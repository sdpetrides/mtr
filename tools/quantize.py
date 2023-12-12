import os
import sys
import datetime

import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx

from eval_utils import eval_utils
from mtr.config import cfg, cfg_from_yaml_file
from mtr.datasets import build_dataloader
from mtr.models import model as model_utils
from mtr.utils import common_utils


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, " \t", "Size (KB):", size / 1e3)
    os.remove("temp.p")
    return size


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)


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

    # print("Model loaded and ready to go!")

    q_context_encoder = torch.quantization.quantize_dynamic(
        model.context_encoder, {nn.Linear}, dtype=torch.qint8
    )

    # m_size = print_size_of_model(model.context_encoder, "fp32")
    # q_size = print_size_of_model(q_context_encoder, "qint8")
    # print("{0:.2f} times smaller".format(m_size / q_size))

    _, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=batch_size,
        dist=dist_test,
        workers=2,
        logger=logger,
        training=False,
    )

    for ce in [model.context_encoder, q_context_encoder]:
        with torch.no_grad():
            model.context_encoder = ce
            print_size_of_model(model)
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

# 263538.428
# 248300.956

# 0.1837
# 0.2014
