import subprocess
import sys


def test_retraining_help_is_available_without_starting_training():
    result = subprocess.run([sys.executable, '-m', 'thermis.retrain', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert 'prepare' in result.stdout
    assert 'train' in result.stdout


def test_training_refuses_production_destination(tmp_path):
    import pytest

    from thermis.retrain import validate_training_output
    with pytest.raises(ValueError, match='runs'):
        validate_training_output(tmp_path/'models'/'tabular')
