import sys
import datetime

import torch

from eval_utils import eval_utils
from mtr.config import cfg, cfg_from_yaml_file
from mtr.datasets import build_dataloader
from mtr.models import model as model_utils
from mtr.utils import common_utils


def main():
    path_to_checkpoint = sys.argv[1]
    cfg_file = sys.argv[2]
    cfg_from_yaml_file(cfg_file, cfg)
    checkpoint = torch.load(path_to_checkpoint)

    output_dir = cfg.ROOT_DIR / "output" / "quantize"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / "eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / (
        "log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    model = model_utils.MotionTransformer(config=cfg.MODEL)
    model.load_state_dict(checkpoint["model_state"])

    batch_size = 4
    dist_test = False

    print("Model loaded and ready to go!")

    _, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=batch_size,
        dist=dist_test,
        workers=2,
        logger=logger,
        training=False,
    )

    with torch.no_grad():
        model.cuda()
        eval_utils.eval_one_epoch(
            cfg,
            model,
            test_loader,
            0,
            logger,
            dist_test=dist_test,
            result_dir=eval_output_dir,
            save_to_file=False,
        )


if __name__ == "__main__":
    main()
