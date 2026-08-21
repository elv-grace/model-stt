import setproctitle
from dacite import from_dict

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default
from common_ml.utils import nested_update

from config import config
from src.model import RuntimeConfig, WhisperSTT
from src.punctuate import PunctuationConfig

# DISABLED (translation): path C's translator config
# from src.translate import TranslatorConfig

if __name__ == '__main__':
    setproctitle.setproctitle('model-whisper-stt')
    catch_errors()

    # user params override the default runtime profile; named profiles in
    # config.yml can be selected with {"profile": "..."} and then further
    # overridden field by field
    params = get_params()
    profile = params.pop("profile", "default")
    defaults = config["runtime"].get(profile)
    if defaults is None:
        raise ValueError(
            f"unknown profile {profile!r}; known: {sorted(config['runtime'])}"
        )
    merged = nested_update(nested_update(config["runtime"]["default"], defaults), params)
    cfg = from_dict(RuntimeConfig, merged)

    model = WhisperSTT(
        cfg,
        models=config["models"],
        weights_dir=config["storage"]["weights_dir"],
        sentence_gap_ms=config["postprocessing"]["sentence_gap"],
        max_caption_words=config["postprocessing"]["max_caption_words"],
        punctuation=from_dict(
            PunctuationConfig, config["postprocessing"].get("punctuation", {})
        ),
        # DISABLED (translation): translate_fallback / translator_cfg
    )

    run_default(model)
