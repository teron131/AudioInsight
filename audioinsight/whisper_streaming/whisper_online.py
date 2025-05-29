#!/usr/bin/env python3
import logging
import sys
import time
from functools import lru_cache

from .backends import FasterWhisperASR, OpenAIAPIASR
from .online_asr import OnlineASRProcessor, VACOnlineASRProcessor

logger = logging.getLogger(__name__)


WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")


def create_tokenizer(lang):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert lang in WHISPER_LANG_CODES or lang == "auto", "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    # Chinese tokenizer doesn't work properly with Chinese punctuation (。！？),
    # so skip sentence tokenization for Chinese languages
    if lang in ["zh", "zh-cn", "zh-tw"]:
        return None

    # supported by fast-mosestokenizer
    if lang in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue".split() or lang == "auto":
        from mosestokenizer import MosesSentenceSplitter

        return MosesSentenceSplitter(lang)

    else:
        return None


def backend_factory(args):
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenAIAPIASR(lang=args.lang)
    else:
        asr_cls = FasterWhisperASR

        # Only for FasterWhisperASR and WhisperTimestampedASR
        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model for language {args.lang}...")
        asr = asr_cls(
            modelsize=size,
            lang=args.lang,
            cache_dir=args.model_cache_dir,
            model_dir=args.model_dir,
        )
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

    # Apply common configurations
    if getattr(args, "vad", False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lang
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":

        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None
    return asr, tokenizer


def online_factory(args, asr, tokenizer, logfile=sys.stderr):
    if args.vac:
        online = VACOnlineASRProcessor(args.min_chunk_size, asr, tokenizer, logfile=logfile, buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec), confidence_validation=args.confidence_validation)
    else:
        online = OnlineASRProcessor(asr, tokenizer, logfile=logfile, buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec), confidence_validation=args.confidence_validation)
    return online


def asr_factory(args, logfile=sys.stderr):
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    asr, tokenizer = backend_factory(args)
    online = online_factory(args, asr, tokenizer, logfile=logfile)
    return asr, online


def warmup_asr(asr, warmup_file=None, timeout=5):
    """
    Warmup the ASR model by transcribing a short audio file.
    """
    import os
    import tempfile

    if warmup_file is None:
        # Download JFK sample if not already present
        jfk_url = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
        temp_dir = tempfile.gettempdir()
        warmup_file = os.path.join(temp_dir, "whisper_warmup_jfk.wav")

        if not os.path.exists(warmup_file):
            logger.debug(f"Downloading warmup file from {jfk_url}")
            print(f"Downloading warmup file from {jfk_url}")
            import socket
            import time
            import urllib.error
            import urllib.request

            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)

            start_time = time.time()
            try:
                urllib.request.urlretrieve(jfk_url, warmup_file)
                logger.debug(f"Download successful in {time.time() - start_time:.2f}s")
            except (urllib.error.URLError, socket.timeout) as e:
                logger.warning(f"Download failed: {e}. Proceeding without warmup.")
                return False
            finally:
                socket.setdefaulttimeout(original_timeout)
    elif not warmup_file:
        return False

    if not warmup_file or not os.path.exists(warmup_file) or os.path.getsize(warmup_file) == 0:
        logger.warning(f"Warmup file {warmup_file} invalid or missing.")
        return False

    print(f"Warming up Whisper with {warmup_file}")
    try:
        import librosa

        audio, sr = librosa.load(warmup_file, sr=16000)
    except Exception as e:
        logger.warning(f"Failed to load audio file: {e}")
        return False

    # Process the audio
    asr.transcribe(audio)

    logger.info("Whisper is warmed up")
