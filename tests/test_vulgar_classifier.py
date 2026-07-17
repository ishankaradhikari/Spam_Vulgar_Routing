from pathlib import Path

from vulgar_classifier import VulgarClassifier


DATASET = Path(__file__).resolve().parent.parent / 'dataset' / 'vulgar_dataset.txt'


def test_clean_negative_examples_remain_clean():
    model = VulgarClassifier(alpha=2.0, vulgar_threshold=0.80)
    model.train(str(DATASET))

    clean_examples = [
        'You are being ridiculous today.',
        'That idea makes absolutely no sense.',
        'I cannot believe you did that again.',
        'You are annoying.',
        'You clearly have no idea what you are doing.',
    ]

    for text in clean_examples:
        label, prob = model.predict(text)
        assert label == model.CLEAN, f'{text} should stay clean, got {label} ({prob:.3f})'


def test_profane_examples_are_flagged():
    model = VulgarClassifier(alpha=2.0, vulgar_threshold=0.80)
    model.train(str(DATASET))

    vulgar_examples = [
        'fuck you',
        'fucking idiot',
        'bullshit',
        'asshole',
        'bitch',
        'motherfucker',
        'stfu',
        'wtf',
        'fk off',
        'fck that',
    ]

    for text in vulgar_examples:
        label, prob = model.predict(text)
        assert label == model.VULGAR, f'{text} should be vulgar, got {label} ({prob:.3f})'


def test_model_can_be_saved_and_loaded(tmp_path):
    model = VulgarClassifier(alpha=2.0, vulgar_threshold=0.80)
    model.train(str(DATASET))

    save_path = tmp_path / 'vulgar_model.pkl'
    model.save(str(save_path))

    loaded = VulgarClassifier.load(str(save_path))

    assert loaded.trained is True
    assert loaded.class_counts[loaded.VULGAR] == model.class_counts[model.VULGAR]

    label, prob = loaded.predict('fuck you')
    assert label == loaded.VULGAR, f'loaded model should flag vulgar text, got {label} ({prob:.3f})'
