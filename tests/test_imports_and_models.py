import sys
import unittest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ImportAndModelTests(unittest.TestCase):
    def test_public_package_imports(self) -> None:
        from speedrunning_plms import PLM, PLMConfig
        from speedrunning_plms.data import ChunkPacker, LegacyFlatPacker, TokenIds, read_shard_tokens
        from speedrunning_plms.flex import generate_dilated_sliding_window
        from speedrunning_plms.optim import Muon

        self.assertIsNotNone(PLM)
        self.assertIsNotNone(PLMConfig)
        self.assertIsNotNone(ChunkPacker)
        self.assertIsNotNone(LegacyFlatPacker)
        self.assertIsNotNone(TokenIds)
        self.assertIsNotNone(read_shard_tokens)
        self.assertIsNotNone(generate_dilated_sliding_window)
        self.assertIsNotNone(Muon)

    def test_root_compatibility_imports(self) -> None:
        from data.dataloading import EvalLoader
        from model.model import PLM, PLMConfig
        from optimizer import Muon

        self.assertIsNotNone(EvalLoader)
        self.assertIsNotNone(PLM)
        self.assertIsNotNone(PLMConfig)
        self.assertIsNotNone(Muon)

    def test_model_explicit_token_ids_avoid_tokenizer_requirement(self) -> None:
        from speedrunning_plms.models import PLM, PLMConfig

        config = PLMConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=2,
            vocab_size=33,
            unet=False,
            compile_flex_attention=False,
            tokenizer_name=None,
            cls_token_id=0,
            eos_token_id=2,
            pad_token_id=1,
            mask_token_id=32,
        )
        model = PLM(config)
        self.assertIsNone(model.tokenizer)
        self.assertIn("embedding.weight", model.state_dict())


if __name__ == "__main__":
    unittest.main()
