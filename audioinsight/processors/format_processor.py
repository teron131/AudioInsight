from .base_processor import (
    BaseProcessor,
    _sentence_split_regex,
    format_time,
    logger,
    s2hk,
)


class FormatProcessor(BaseProcessor):
    """Handles formatting of transcription and diarization results."""

    def __init__(self, args):
        super().__init__(args)

    async def format_by_sentences(self, tokens, sep, end_attributed_speaker, online=None):
        """Format tokens by sentence boundaries using the sentence tokenizer."""
        if not tokens:
            return []

        # Build full text from all tokens - optimize by filtering first
        # REWORK: Use parsed_text if available, prioritize parsed tokens
        token_texts = []
        for token in tokens:
            # If this is a parsed token, use its text directly (it represents the full parsed transcript)
            if hasattr(token, "is_parsed") and token.is_parsed:
                token_texts = [token.text]  # Use only the parsed text, ignore other tokens
                break
            else:
                # Use parsed_text if available, otherwise use original text
                text = token.parsed_text or token.text
                if text and text.strip():
                    token_texts.append(text)

        if not token_texts:
            return []

        full_text = sep.join(token_texts)

        try:
            # Use the sentence tokenizer to split into sentences
            if online and hasattr(online, "tokenize") and online.tokenize:
                try:
                    # MosesSentenceSplitter expects a list input
                    sentence_texts = online.tokenize([full_text])
                except Exception as e:
                    # Fallback for other tokenizers that might expect string input
                    try:
                        sentence_texts = online.tokenize(full_text)
                    except Exception as e2:
                        logger.warning(f"Sentence tokenization failed: {e2}. Falling back to speaker-based segmentation.")
                        return await self.format_by_speaker(tokens, sep, end_attributed_speaker)
            else:
                # No tokenizer, split by basic punctuation
                sentence_texts = _sentence_split_regex.split(full_text)
                sentence_texts = [s.strip() for s in sentence_texts if s.strip()]

            if not sentence_texts:
                sentence_texts = [full_text]

            # Map sentences back to tokens and create lines
            lines = []
            token_index = 0

            for sent_text in sentence_texts:
                sent_text = sent_text.strip()
                if not sent_text:
                    continue

                # Find tokens that make up this sentence
                sent_tokens = []
                accumulated = ""
                start_token_index = token_index

                # Accumulate tokens until we roughly match the sentence text
                while token_index < len(tokens) and len(accumulated) < len(sent_text):
                    token = tokens[token_index]
                    if token.text.strip():  # Only consider non-empty tokens
                        accumulated = (accumulated + " " + token.text).strip() if accumulated else token.text
                        sent_tokens.append(token)
                    token_index += 1

                # If we didn't get any tokens, try to get at least one
                if not sent_tokens and start_token_index < len(tokens):
                    sent_tokens = [tokens[start_token_index]]
                    token_index = start_token_index + 1

                if sent_tokens:
                    # Determine speaker (use most common speaker in the sentence) - optimize speaker detection
                    if self.args.diarization:
                        # Filter valid speakers once
                        valid_speakers = [t.speaker for t in sent_tokens if t.speaker not in {-1} and t.speaker is not None]
                        if valid_speakers:
                            # Use most frequent speaker with optimized counting
                            speaker = max(set(valid_speakers), key=valid_speakers.count)
                        else:
                            speaker = sent_tokens[0].speaker
                    else:
                        speaker = 0  # Default speaker when no diarization (UI will show "Speaker 1")

                    # Create line for this sentence
                    line = {"speaker": speaker, "text": sent_text, "beg": format_time(sent_tokens[0].start), "end": format_time(sent_tokens[-1].end), "diff": 0}  # Not used in sentence mode
                    lines.append(line)

            return lines

        except Exception as e:
            logger.warning(f"Error in sentence-based formatting: {e}. Falling back to speaker-based segmentation.")
            return await self.format_by_speaker(tokens, sep, end_attributed_speaker)

    async def format_by_speaker(self, tokens, sep, end_attributed_speaker):
        """Format tokens by speaker changes (original behavior)."""
        previous_speaker = -1
        lines = []
        last_end_diarized = 0
        undiarized_text = []

        # Check if we have a parsed token that represents the entire transcript
        parsed_token = None
        for token in tokens:
            if hasattr(token, "is_parsed") and token.is_parsed:
                parsed_token = token
                break

        if parsed_token:
            # If we have a parsed token, create a single line with the parsed text
            lines.append({"speaker": parsed_token.speaker, "text": parsed_token.text, "beg": format_time(parsed_token.start), "end": format_time(parsed_token.end), "diff": 0})
            return lines

        # Process each token (original behavior for non-parsed tokens)
        for token in tokens:
            speaker = token.speaker

            # Handle diarization - optimize with set membership checks
            if self.args.diarization:
                speaker_in_invalid_set = speaker in {-1} or speaker is None
                if speaker_in_invalid_set and token.end >= end_attributed_speaker:
                    # Keep collecting undiarized text but also display it temporarily
                    undiarized_text.append(token.text)
                    # Assign a temporary positive speaker ID for display purposes
                    speaker = 0  # Use Speaker 0 as default for undiarized tokens (UI will show "Speaker 1")
                elif speaker_in_invalid_set and token.end < end_attributed_speaker:
                    speaker = previous_speaker if previous_speaker >= 0 else 0
                if not speaker_in_invalid_set:
                    last_end_diarized = max(token.end, last_end_diarized)

            # Group by speaker
            if speaker != previous_speaker or not lines:
                # REWORK: Use parsed_text if available
                text_to_add = token.parsed_text or token.text
                lines.append({"speaker": speaker, "text": text_to_add, "beg": format_time(token.start), "end": format_time(token.end), "diff": round(token.end - last_end_diarized, 2)})
                previous_speaker = speaker
            else:
                text_to_add = token.parsed_text or token.text
                if text_to_add:  # Only append if text isn't empty
                    # Append token text directly - duplication prevented by work coordination
                    lines[-1]["text"] += sep + text_to_add
                lines[-1]["end"] = format_time(token.end)
                lines[-1]["diff"] = round(token.end - last_end_diarized, 2)

        return lines
