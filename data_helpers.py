import os
import urllib.request
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

from sklearn.metrics import classification_report


def download_file(url, output_path):
    """Download a single file from URL to output_path."""
    try:
        if not os.path.exists(output_path):
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ Downloaded: {os.path.basename(output_path)}")
        else:
            print(f"✓ Already exists: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {os.path.basename(output_path)}: {e}")
        return False


def download_dataset_files(base_url, file_list, output_dir='ud_data'):
    """Download multiple files from a base URL to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    for file in file_list:
        url = base_url + file
        output_path = os.path.join(output_dir, file)
        if download_file(url, output_path):
            success_count += 1

    return success_count


def download_all_datasets():
    """Download Czech and English UD datasets."""
    base_url_czech = "https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-PDTC/master/"
    base_url_english = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master/"

    czech_files = [
        "cs_pdtc-ud-train-la.conllu",
        "cs_pdtc-ud-train-ca.conllu",
        "cs_pdtc-ud-dev.conllu",
        "cs_pdtc-ud-test.conllu"
    ]

    english_files = [
        "en_gum-ud-train.conllu",
        "en_gum-ud-dev.conllu",
        "en_gum-ud-test.conllu"
    ]

    czech_count = download_dataset_files(base_url_czech, czech_files)
    english_count = download_dataset_files(base_url_english, english_files)

    print(f"\nTotal: {czech_count}/{len(czech_files)} Czech files, {english_count}/{len(english_files)} English files")


def is_valid_line(line):
    line = line.strip()
    if not line:
        return False
    if line.startswith('#'):
        return False
    columns = line.split('\t')
    if columns:
        first_column = columns[0]
        if '-' in first_column or '.' in first_column:
            return False
    return True

def count_valid_lines(file_path):
    valid_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if is_valid_line(line):
                valid_count += 1
    return valid_count

def filter_valid_lines(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            stripped = line.strip()
            if not stripped:
                f_out.write(line)
                continue
            if stripped.startswith('#'):
                f_out.write(line)
                continue
            columns = stripped.split('\t')
            if columns:
                first_column = columns[0]
                if '-' in first_column or '.' in first_column:
                    continue
            f_out.write(line)

def truncate_to_n_sentences(input_path, output_path, n_sentences):
    """Truncate a CoNLL-U file to n sentences"""
    sentence_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            stripped = line.strip()

            # Write the line
            f_out.write(line)

            # Count sentences (empty lines separate sentences)
            if not stripped:
                sentence_count += 1
                if sentence_count >= n_sentences:
                    break

def count_word_tag_pairs(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if is_valid_line(line):
                count += 1
    return count


def read_conllu_sentences(file_path):
    """Read CoNLL-U file and return list of sentences with tokens and gold tags"""
    sentences = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()

            # Empty line marks end of sentence
            if not stripped:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # Skip comments
            if stripped.startswith('#'):
                continue

            # Parse token line
            columns = stripped.split('\t')
            if len(columns) >= 10:
                token_id = columns[0]
                # Skip multi-word tokens and ellipsis
                if '-' in token_id or '.' in token_id:
                    continue

                word = columns[1]
                gold_tag = columns[3]  # UPOS tag
                current_sentence.append({'word': word, 'gold_tag': gold_tag})

        # Add last sentence if exists
        if current_sentence:
            sentences.append(current_sentence)

    return sentences

def print_confusion_matrix(gold, pred, language):
    tags = sorted(set(gold + pred))

    cm = confusion_matrix(gold, pred, labels=tags)

    print(f"\n{language} Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=tags, columns=tags)
    return cm_df

def compute_lambdas(p0, p1, p2, p3, held_out_words, vocab_size, epsilon=0.0001, max_iter=10_000):
    lambdas = [0.25, 0.25, 0.25, 0.25]

    for iteration in range(max_iter):
        c = [0.0, 0.0, 0.0, 0.0]

        for i in range(2, len(held_out_words)):
            word = held_out_words[i]
            history_bigram = (held_out_words[i-2], held_out_words[i-1])

            prob_0 = p0

            prob_1 = p1.get(word, 0)


            bigram_for_p2 = (history_bigram[1], word)
            if history_bigram[1] in p1 and p1[history_bigram[1]] > 0:
                prob_2 = p2.get(bigram_for_p2, 0)
            else:
                prob_2 = 1 / vocab_size


            trigram = (history_bigram[0], history_bigram[1], word)
            if history_bigram in p2 and p2[history_bigram] > 0:
                prob_3 = p3.get(trigram, 0)
            else:
                prob_3 = 1 / vocab_size

            p_interp = (lambdas[0] * prob_0 +
                       lambdas[1] * prob_1 +
                       lambdas[2] * prob_2 +
                       lambdas[3] * prob_3)

            if p_interp > 0:
                c[0] += (lambdas[0] * prob_0) / p_interp
                c[1] += (lambdas[1] * prob_1) / p_interp
                c[2] += (lambdas[2] * prob_2) / p_interp
                c[3] += (lambdas[3] * prob_3) / p_interp

        total_c = sum(c)
        if total_c > 0:
            new_lambdas = [c[j] / total_c for j in range(4)]
        else:
            new_lambdas = lambdas[:]

        converged = True
        for j in range(4):
            if abs(lambdas[j] - new_lambdas[j]) >= epsilon:
                converged = False
                break

        lambdas = new_lambdas

        if converged:
            print(f"Converged after {iteration + 1} iterations")
            break

    return lambdas

def read_conllu_word_tag_sentences(file_path, tag_column=3):
    sentences = []
    current = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Sentence boundary
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue

            # Skip comments
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                continue

            token_id = cols[0]
            if "-" in token_id or "." in token_id:
                continue

            word = cols[1]
            tag = cols[tag_column]
            current.append((word, tag))

    if current:
        sentences.append(current)

    return sentences

def print_per_tag_performance(gold, pred, language):
    """Print per-tag precision, recall, and F1-score"""
    print(f"\n{language} Per-Tag Performance:")
    print("=" * 60)

    # Generate classification report
    report = classification_report(gold, pred, digits=4, zero_division=0)
    print(report)

    # Also create DataFrame for better analysis
    from sklearn.metrics import precision_recall_fscore_support

    tags = sorted(set(gold + pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=tags, zero_division=0
    )

    # Create DataFrame
    df = pd.DataFrame({
        'Tag': tags,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

    # Sort by F1-score to see best/worst performers
    df_sorted = df.sort_values('F1-Score', ascending=False)

    print(f"\n{language} Tags Sorted by F1-Score:")
    print(df_sorted.to_string(index=False))

    return df