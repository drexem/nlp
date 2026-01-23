import data_helpers
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from collections import Counter, defaultdict
import math
from tqdm import tqdm

import data_helpers

def train_hmm_counts(sentences):
    tag_unigrams = Counter()
    tag_bigrams = Counter()
    tag_trigrams = Counter()

    word_unigrams = Counter()
    tag_word = Counter()

    for sent in sentences:
        tags = [t for w, t in sent]
        for w, t in sent:
            word_unigrams[w] += 1
            tag_word[(t, w)] += 1
        tags_ext = ["<s>", "<s>"] + tags + ["</s>"]

        for t in tags_ext[2:]:
            tag_unigrams[t] += 1

        for i in range(2, len(tags_ext)):
            t_im2 = tags_ext[i - 2]
            t_im1 = tags_ext[i - 1]
            t_i = tags_ext[i]

            tag_bigrams[(t_im1, t_i)] += 1
            tag_trigrams[(t_im2, t_im1, t_i)] += 1

    return tag_unigrams, tag_bigrams, tag_trigrams, word_unigrams, tag_word

def get_tag_ngram_probs(tag_vocab_size, tag_unigrams, tag_bigrams, tag_trigrams, train_tag_tokens):
    p0 = 1 / tag_vocab_size

    p1 = {}
    for t in tag_unigrams:
        p1[t] = tag_unigrams[t] / train_tag_tokens

    p2 = {}
    for (t_prev, t) in tag_bigrams:
        if tag_unigrams[t_prev] > 0:
            p2[(t_prev, t)] = tag_bigrams[(t_prev, t)] / tag_unigrams[t_prev]

    p3 = {}
    for (t2, t1, t) in tag_trigrams:
        if tag_bigrams[(t2, t1)] > 0:
            p3[(t2, t1, t)] = tag_trigrams[(t2, t1, t)] / tag_bigrams[(t2, t1)]

    return p0, p1, p2, p3

def get_emission_probs(tag_word, tag_unigrams, word_unigrams, vocab_words):
    total_words = sum(word_unigrams.values())
    V = len(vocab_words)

    p0_word = 1 / V

    p1_word = {}
    for w in word_unigrams:
        p1_word[w] = word_unigrams[w] / total_words

    p2_emit = {}
    for (t, w), c in tag_word.items():
        if tag_unigrams[t] > 0:
            p2_emit[(t, w)] = c / tag_unigrams[t]

    return p0_word, p1_word, p2_emit


def emission_prob(word, tag, p0_word, p1_word, p2_emit, mu, V):
    p0 = p0_word
    p1 = p1_word.get(word, 0.0)
    p2 = p2_emit.get((tag, word), 0.0)

    return mu[0]*p0 + mu[1]*p1 + mu[2]*p2

def transition_prob(t_im2, t_im1, t_i, p0, p1, p2, p3, lambdas):
    prob_0 = p0
    prob_1 = p1.get(t_i, 0.0)
    prob_2 = p2.get((t_im1, t_i), 0.0)
    prob_3 = p3.get((t_im2, t_im1, t_i), 0.0)

    return (lambdas[0]*prob_0 +
            lambdas[1]*prob_1 +
            lambdas[2]*prob_2 +
            lambdas[3]*prob_3)

def viterbi_decode(words, tags_set, p_tag0, p_tag1, p_tag2, p_tag3, lambdas,
                   p0_word, p1_word, p2_emit, mu, tag_vocab_size, word_vocab_size):

    NEG_INF = -1e18

    dp = {("<s>", "<s>"): 0.0}
    bp = {}

    for i, w in enumerate(words):
        new_dp = defaultdict(lambda: NEG_INF)
        new_bp = {}

        for (t_im2, t_im1), prev_score in dp.items():
            if prev_score <= NEG_INF / 2:
                continue

            for t_i in tags_set:
                e = emission_prob(w, t_i, p0_word, p1_word, p2_emit, mu, word_vocab_size)
                if e <= 0:
                    continue
                log_e = math.log(e)

                tr = transition_prob(t_im2, t_im1, t_i, p_tag0, p_tag1, p_tag2, p_tag3, lambdas)
                if tr <= 0:
                    continue
                log_tr = math.log(tr)

                score = prev_score + log_tr + log_e

                pair = (t_im1, t_i)
                if score > new_dp[pair]:
                    new_dp[pair] = score
                    new_bp[pair] = (t_im2, t_im1)

        dp = new_dp
        bp[i] = new_bp

    # add transition to </s>
    best_score = NEG_INF
    best_pair = None

    for (t_im2, t_im1), score in dp.items():
        tr = transition_prob(t_im2, t_im1, "</s>", p_tag0, p_tag1, p_tag2, p_tag3, lambdas)
        if tr <= 0:
            continue
        score_end = score + math.log(tr)

        if score_end > best_score:
            best_score = score_end
            best_pair = (t_im2, t_im1)

    n = len(words)
    if best_pair is None:
        return ["NOUN"] * n

    # backtrack
    tags_out = [None] * n
    t_im2, t_im1 = best_pair

    if n == 1:
        tags_out[0] = t_im1
        return tags_out

    tags_out[n - 1] = t_im1
    tags_out[n - 2] = t_im2

    cur_pair = (t_im2, t_im1)

    for i in range(n - 1, 1, -1):
        prev_pair = bp[i].get(cur_pair, None)
        if prev_pair is None:
            break
        tags_out[i - 2] = prev_pair[0]
        cur_pair = prev_pair

    for i in range(n):
        if tags_out[i] is None:
            tags_out[i] = "NOUN"

    return tags_out

def flatten_tags(sentences):
    return [t for sent in sentences for (w, t) in sent]

def tune_emission_mus(train_sentences, dev_sentences):
    tag_unigrams, tag_bigrams, tag_trigrams, word_unigrams, tag_word = train_hmm_counts(train_sentences)
    vocab_words = set(word_unigrams.keys())
    p0_word, p1_word, p2_emit = get_emission_probs(tag_word, tag_unigrams, word_unigrams, vocab_words)
    mu = [1/3, 1/3, 1/3]
    heldout_pairs = [(w, t) for sent in dev_sentences for (w, t) in sent]

    eps = 1e-4
    max_iter = 200

    for it in range(max_iter):
        c = [0.0, 0.0, 0.0]

        for w, t in heldout_pairs:
            p0 = p0_word
            p1 = p1_word.get(w, 0.0)
            p2 = p2_emit.get((t, w), 0.0)

            p = mu[0]*p0 + mu[1]*p1 + mu[2]*p2
            if p <= 0:
                continue

            c[0] += (mu[0]*p0) / p
            c[1] += (mu[1]*p1) / p
            c[2] += (mu[2]*p2) / p

        s = sum(c)
        new_mu = [x/s for x in c]

        if max(abs(new_mu[i] - mu[i]) for i in range(3)) < eps:
            break
        mu = new_mu

    return mu


def train_and_eval_hmm(train_path, dev_path, test_path, language_name="LANG"):
    print(f"\n{'='*70}")
    print(f"Training HMM tagger for {language_name}")

    train_sents = data_helpers.read_conllu_word_tag_sentences(train_path)
    dev_sents = data_helpers.read_conllu_word_tag_sentences(dev_path)
    test_sents = data_helpers.read_conllu_word_tag_sentences(test_path)

    tag_unigrams, tag_bigrams, tag_trigrams, word_unigrams, tag_word = train_hmm_counts(train_sents)

    tags_set = sorted(set(tag_unigrams.keys()) - {"<s>", "</s>"})
    tag_vocab_size = len(tags_set) + 2
    train_tag_tokens = sum(tag_unigrams.values())

    p_tag0, p_tag1, p_tag2, p_tag3 = get_tag_ngram_probs(
        tag_vocab_size, tag_unigrams, tag_bigrams, tag_trigrams, train_tag_tokens
    )

    dev_tags_seq = ["<s>", "<s>"] + flatten_tags(dev_sents) + ["</s>"]
    lambdas = data_helpers.compute_lambdas(
        p_tag0, p_tag1, p_tag2, p_tag3,
        dev_tags_seq,
        vocab_size=tag_vocab_size,
        epsilon=1e-4,
        max_iter=2000
    )
    mu = tune_emission_mus(train_sents, dev_sents)

    vocab_words = set(word_unigrams.keys())
    word_vocab_size = len(vocab_words) + 1
    p0_word, p1_word, p2_emit = get_emission_probs(tag_word, tag_unigrams, word_unigrams, vocab_words)

    gold_all = []
    pred_all = []

    print(f"Decoding {language_name} test set with Viterbi...")
    for sent in tqdm(test_sents):
        words = [w for w, t in sent]
        gold_tags = [t for w, t in sent]

        pred_tags = viterbi_decode(
            words, tags_set,
            p_tag0, p_tag1, p_tag2, p_tag3, lambdas,
            p0_word, p1_word, p2_emit, mu,
            tag_vocab_size, word_vocab_size
        )

        gold_all.extend(gold_tags)
        pred_all.extend(pred_tags)

    acc = accuracy_score(gold_all, pred_all)
    print(f"[{language_name}] HMM Accuracy: {acc} / ({acc*100}%)")

    tags_sorted = sorted(set(gold_all + pred_all))
    cm = confusion_matrix(gold_all, pred_all, labels=tags_sorted)
    cm_df = pd.DataFrame(cm, index=tags_sorted, columns=tags_sorted)

    return acc, cm_df

def split_supervised_unsupervised(train_sents, n_supervised_pairs=10_000):
    sup = []
    unsup = []

    count = 0
    for sent in train_sents:
        if count >= n_supervised_pairs:
            unsup.append([w for (w, t) in sent])
            continue
        if count + len(sent) <= n_supervised_pairs:
            sup.append(sent)
            count += len(sent)
        else:
            k = n_supervised_pairs - count
            sup.append(sent[:k])
            count += k
            rest_words = [w for (w, t) in sent[k:]]
            if rest_words:
                unsup.append(rest_words)

    return sup, unsup

def build_initial_hmm_from_supervised(supervised_sents, dev_sents):
    tag_unigrams, tag_bigrams, tag_trigrams, word_unigrams, tag_word = train_hmm_counts(supervised_sents)

    tags_set = sorted(set(tag_unigrams.keys()) - {"<s>", "</s>"})
    tag_vocab_size = len(tags_set) + 2
    train_tag_tokens = sum(tag_unigrams.values())

    p_tag0, p_tag1, p_tag2, p_tag3 = get_tag_ngram_probs(
        tag_vocab_size, tag_unigrams, tag_bigrams, tag_trigrams, train_tag_tokens
    )

    dev_tags_seq = ["<s>", "<s>"] + flatten_tags(dev_sents) + ["</s>"]
    lambdas = data_helpers.compute_lambdas(
        p_tag0, p_tag1, p_tag2, p_tag3,
        dev_tags_seq,
        vocab_size=tag_vocab_size,
        epsilon=1e-4,
        max_iter=2000
    )

    mu = tune_emission_mus(supervised_sents, dev_sents)

    vocab_words = set(word_unigrams.keys())
    word_vocab_size = len(vocab_words) + 1
    p0_word, p1_word, p2_emit = get_emission_probs(tag_word, tag_unigrams, word_unigrams, vocab_words)

    model = {
        "tags_set": tags_set,
        "tag_vocab_size": tag_vocab_size,
        "word_vocab_size": word_vocab_size,
        "lambdas": lambdas,
        "mu": mu,
        "p_tag0": p_tag0,
        "p_tag1": p_tag1,
        "p_tag2": p_tag2,
        "p_tag3": p_tag3,
        "p0_word": p0_word,
        "p1_word": p1_word,
        "p2_emit": p2_emit,
    }
    return model

def forward_backward_trigram(words, tags_set, model):
    p_tag0 = model["p_tag0"]
    p_tag1 = model["p_tag1"]
    p_tag2 = model["p_tag2"]
    p_tag3 = model["p_tag3"]
    lambdas = model["lambdas"]

    p0_word = model["p0_word"]
    p1_word = model["p1_word"]
    p2_emit = model["p2_emit"]
    mu = model["mu"]

    word_vocab_size = model["word_vocab_size"]

    T = len(words)
    assert T > 0

    alpha = [defaultdict(float) for _ in range(T)]
    beta  = [defaultdict(float) for _ in range(T)]
    scales = [1.0 for _ in range(T)]

    w0 = words[0]
    for t0 in tags_set:
        tr = transition_prob("<s>", "<s>", t0,
                             p_tag0, p_tag1, p_tag2, p_tag3, lambdas)
        em = emission_prob(w0, t0, p0_word, p1_word, p2_emit, mu, word_vocab_size)
        alpha[0][("<s>", t0)] = tr * em

    s0 = sum(alpha[0].values())
    if s0 == 0.0:
        s0 = 1e-300
    scales[0] = s0
    for st in alpha[0]:
        alpha[0][st] /= scales[0]

    for i in range(1, T):
        wi = words[i]
        for (t_im2, t_im1), a_prev in alpha[i-1].items():
            if a_prev == 0.0:
                continue
            for t_i in tags_set:
                tr = transition_prob(t_im2, t_im1, t_i,
                                     p_tag0, p_tag1, p_tag2, p_tag3, lambdas)
                if tr == 0.0:
                    continue
                em = emission_prob(wi, t_i, p0_word, p1_word, p2_emit, mu, word_vocab_size)
                if em == 0.0:
                    continue
                alpha[i][(t_im1, t_i)] += a_prev * tr * em

        si = sum(alpha[i].values())
        if si == 0.0:
            si = 1e-300
        scales[i] = si
        for st in alpha[i]:
            alpha[i][st] /= scales[i]

    logZ = sum(math.log(s) for s in scales)

    for (t_im1, t_i) in alpha[T-1].keys():
        beta[T-1][(t_im1, t_i)] = transition_prob(
            t_im1, t_i, "</s>",
            p_tag0, p_tag1, p_tag2, p_tag3, lambdas
        )

    for st in beta[T-1]:
        beta[T-1][st] /= scales[T-1]

    for i in range(T-2, -1, -1):
        w_next = words[i+1]
        for (t_im2, t_im1) in alpha[i].keys():
            s = 0.0
            for t_i in tags_set:
                tr = transition_prob(t_im2, t_im1, t_i,
                                     p_tag0, p_tag1, p_tag2, p_tag3, lambdas)
                if tr == 0.0:
                    continue
                em = emission_prob(w_next, t_i, p0_word, p1_word, p2_emit, mu, word_vocab_size)
                if em == 0.0:
                    continue
                s += tr * em * beta[i+1].get((t_im1, t_i), 0.0)

            beta[i][(t_im2, t_im1)] = s

        for st in beta[i]:
            beta[i][st] /= scales[i]

    return alpha, beta, scales, logZ

def baum_welch_full_trigram(unlabeled_sents_words, model, n_iters=5, min_prob=1e-12):
    tags_set = model["tags_set"]
    tag_vocab_size = model["tag_vocab_size"]
    word_vocab_size = model["word_vocab_size"]

    for it in range(n_iters):
        print(f"\n[Full Baum-Welch] Iteration {it+1}/{n_iters}")

        exp_tri = Counter()     # c(t_{i-2}, t_{i-1}, t_i)
        exp_bi_hist = Counter() # c(t_{i-2}, t_{i-1}) = sum_{t_i} c(tri)
        exp_tag_word = Counter()  # c(t, w)
        exp_tag = Counter()       # c(t)
        exp_word = Counter()      # unigram word counts

        total_logZ = 0.0
        total_tokens = 0

        for words in tqdm(unlabeled_sents_words):
            if not words:
                continue

            alpha, beta, scales, logZ = forward_backward_trigram(words, tags_set, model)
            total_logZ += logZ
            total_tokens += len(words)

            T = len(words)

            for i, w in enumerate(words):
                denom = 0.0
                tmp = {}

                for state, a_val in alpha[i].items():
                    g = a_val * beta[i].get(state, 0.0)
                    tmp[state] = g
                    denom += g

                if denom == 0.0:
                    continue

                for (t_prev, t_cur), g in tmp.items():
                    g /= denom
                    exp_tag_word[(t_cur, w)] += g
                    exp_tag[t_cur] += g

                exp_word[w] += 1.0

            for i in range(1, T):
                wi = words[i]

                denom = 0.0
                tmp_xi = {}
                for (t_im2, t_im1), a_prev in alpha[i-1].items():
                    if a_prev == 0.0:
                        continue

                    for t_i in tags_set:
                        tr = transition_prob(
                            t_im2, t_im1, t_i,
                            model["p_tag0"], model["p_tag1"], model["p_tag2"], model["p_tag3"],
                            model["lambdas"]
                        )
                        if tr == 0.0:
                            continue

                        em = emission_prob(
                            wi, t_i,
                            model["p0_word"], model["p1_word"], model["p2_emit"],
                            model["mu"], word_vocab_size
                        )
                        if em == 0.0:
                            continue

                        b_next = beta[i].get((t_im1, t_i), 0.0)
                        if b_next == 0.0:
                            continue

                        val = a_prev * tr * em * b_next
                        tmp_xi[(t_im2, t_im1, t_i)] = val
                        denom += val

                if denom == 0.0:
                    continue

                for tri, val in tmp_xi.items():
                    frac = val / denom
                    exp_tri[tri] += frac
                    exp_bi_hist[(tri[0], tri[1])] += frac

            denom_end = 0.0
            tmp_end = {}

            for (t_im2, t_im1), a_last in alpha[T-1].items():
                if a_last == 0.0:
                    continue

                tr_end = transition_prob(
                    t_im2, t_im1, "</s>",
                    model["p_tag0"], model["p_tag1"], model["p_tag2"], model["p_tag3"],
                    model["lambdas"]
                )
                if tr_end == 0.0:
                    continue

                val = a_last * tr_end
                tmp_end[(t_im2, t_im1, "</s>")] = val
                denom_end += val

            if denom_end > 0.0:
                for tri_end, val in tmp_end.items():
                    frac = val / denom_end
                    exp_tri[tri_end] += frac
                    exp_bi_hist[(tri_end[0], tri_end[1])] += frac

        avg_nll = -total_logZ / max(total_tokens, 1)
        print(f"[Full Baum-Welch] avg -log P(words) per token: {avg_nll}")

        new_p3 = {}
        for (t2, t1, t0), c in exp_tri.items():
            hist = (t2, t1)
            denom = exp_bi_hist.get(hist, 0.0)
            if denom > 0.0:
                new_p3[(t2, t1, t0)] = max(c / denom, min_prob)
        new_p2_emit = {}
        for (t, w), c in exp_tag_word.items():
            denom = exp_tag.get(t, 0.0)
            if denom > 0.0:
                new_p2_emit[(t, w)] = max(c / denom, min_prob)

        total_w = sum(exp_word.values())
        if total_w == 0.0:
            total_w = 1e-300
        new_p1_word = {w: max(c / total_w, min_prob) for w, c in exp_word.items()}


        exp_bigram = Counter()
        exp_unigram = Counter()

        for (t2, t1, t0), c in exp_tri.items():
            exp_bigram[(t1, t0)] += c
            exp_unigram[t0] += c

        new_p2 = {}
        for (t_prev, t_cur), c in exp_bigram.items():
            denom = exp_unigram.get(t_prev, 0.0)
            if denom > 0.0:
                new_p2[(t_prev, t_cur)] = max(c / denom, min_prob)

        total_tags = sum(exp_unigram.values())
        if total_tags == 0.0:
            total_tags = 1e-300
        new_p1 = {t: max(c / total_tags, min_prob) for t, c in exp_unigram.items()}

        model["p_tag3"] = new_p3
        model["p_tag2"] = new_p2
        model["p_tag1"] = new_p1
        model["p2_emit"] = new_p2_emit
        model["p1_word"] = new_p1_word
        model["p_tag0"] = 1.0 / max(tag_vocab_size, 1)
        model["p0_word"] = 1.0 / max(len(new_p1_word), 1)
        model["word_vocab_size"] = max(len(new_p1_word), 1)

    return model


def train_and_eval_hmm_semi_supervised(train_path, dev_path, test_path, language_name="LANG",
                                       n_supervised_pairs=10_000, bw_iters=5, unsup_sent_lim=1000000):
    print(f"\n{'=' * 70}")
    print(f"SEMI-SUPERVISED HMM (Baum-Welch) for {language_name}")

    train_sents_full = data_helpers.read_conllu_word_tag_sentences(train_path)
    dev_sents = data_helpers.read_conllu_word_tag_sentences(dev_path)
    test_sents = data_helpers.read_conllu_word_tag_sentences(test_path)

    sup_sents, unsup_words = split_supervised_unsupervised(train_sents_full, n_supervised_pairs)
    unsup_words = unsup_words[:unsup_sent_lim]
    print(f"[{language_name}] Supervised sentences: {len(sup_sents)}")
    print(f"[{language_name}] Unlabeled sentences:   {len(unsup_words)}")

    model = build_initial_hmm_from_supervised(sup_sents, dev_sents)
    model['lambdas'] = [0.001, 0, 0, 0.999]
    model['mu'] = [0.001, 0, 0.999]

    def eval_model(model, label):
        gold_all, pred_all = [], []
        for sent in tqdm(test_sents, desc=f"Decoding {label}"):
            words = [w for w, t in sent]
            gold = [t for w, t in sent]
            pred = viterbi_decode(
                words, model["tags_set"],
                model["p_tag0"], model["p_tag1"], model["p_tag2"], model["p_tag3"], model["lambdas"],
                model["p0_word"], model["p1_word"], model["p2_emit"], model["mu"],
                model["tag_vocab_size"], model["word_vocab_size"]
            )
            gold_all.extend(gold)
            pred_all.extend(pred)
        return accuracy_score(gold_all, pred_all), gold_all, pred_all

    baseline_acc, baseline_gold, baseline_pred = eval_model(model, label=f"{language_name} baseline")
    print(f"\n[{language_name}] Baseline accuracy (10k supervised): {baseline_acc}")


    for i in range(bw_iters):
        model = baum_welch_full_trigram(unsup_words, model, n_iters=1)

    bw_acc, bw_gold, bw_pred = eval_model(model, label=f"{language_name} after BW")
    print(f"\n[{language_name}] After Baum-Welch accuracy: {bw_acc}")

    return baseline_acc, bw_acc, baseline_gold, baseline_pred, bw_gold, bw_pred

