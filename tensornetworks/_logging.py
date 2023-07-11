import os
import time
import logging

import torch

_tb_logger = None
__all__ = ["set_logger", "get_tb_logger"]


def set_logger(
    log_file=None,
    log_console_level="info",
    log_file_level=None,
    use_tb_logger=False,
):
    """Bind the default logger with console and file stream output."""

    def _get_level(level):
        if level.lower() == "debug":
            return logging.DEBUG
        elif level.lower() == "info":
            return logging.INFO
        elif level.lower() == "warning":
            return logging.WARN
        elif level.lower() == "error":
            return logging.ERROR
        elif level.lower() == "critical":
            return logging.CRITICAL
        else:
            msg = (
                "`log_console_level` must be one of {{DEBUG, INFO,"
                " WARNING, ERROR, CRITICAL}}, but got {} instead."
            )
            raise ValueError(msg.format(level.upper()))

    _logger = logging.getLogger()

    # Reset
    for h in _logger.handlers:
        _logger.removeHandler(h)

    rq = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
    log_path = os.path.join(os.getcwd(), "logs")

    ch_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(_get_level(log_console_level))
    ch.setFormatter(ch_formatter)
    _logger.addHandler(ch)

    if log_file is not None:
        print("Log will be saved in '{}'.".format(log_path))
        if not os.path.exists(log_path):
            os.mkdir(log_path)
            print("Create folder 'logs/'")
        log_name = os.path.join(log_path, log_file + "-" + rq + ".log")
        print("Start logging into file {}...".format(log_name))
        fh = logging.FileHandler(log_name, mode="w")
        fh.setLevel(
            logging.DEBUG
            if log_file_level is None
            else _get_level(log_file_level)
        )
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - "
            "%(levelname)s: %(message)s"
        )
        fh.setFormatter(fh_formatter)
        _logger.addHandler(fh)
    _logger.setLevel("DEBUG")

    if use_tb_logger:
        tb_log_path = os.path.join(
            log_path, log_file + "-" + rq + "_tb_logger"
        )
        os.mkdir(tb_log_path)
        init_tb_logger(log_dir=tb_log_path)

    return _logger


def init_tb_logger(log_dir):
    try:
        import tensorboard  # noqa: F401
    except ModuleNotFoundError:
        msg = (
            "Cannot load the module tensorboard. Please make sure that"
            " tensorboard is installed."
        )
        raise ModuleNotFoundError(msg)

    from torch.utils.tensorboard import SummaryWriter

    global _tb_logger

    if not _tb_logger:
        _tb_logger = SummaryWriter(log_dir=log_dir)


def get_tb_logger():
    return _tb_logger



class Summary:
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries),flush=True)
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res