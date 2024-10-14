
def create_ambig_context(text, target_char_index):
    
    # Split the text into sentences
    sentences = text.split(" . ")

    # Variable to track character position
    char_count = 0
    sentence_to_delete = None

    # Iterate over each sentence
    for sentence in sentences:
        # Check if the 199th character falls within this sentence
        if char_count <= target_char_index < char_count + len(sentence):
            sentence_to_delete = sentence
            break
        
        # Update character count to reflect the position within the text
        char_count += len(sentence) + 3  # +1 for the space after the sentence

    # Remove the sentence
    if sentence_to_delete:
        updated_text = text.replace(sentence_to_delete, "").strip()
        return updated_text
    else:
        return ""
