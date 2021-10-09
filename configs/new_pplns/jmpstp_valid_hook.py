from openselfsup.framework.hooks.validate_hook import ValidateHook
import sys


class JumpStop_ValidateHook(ValidateHook):
    def __init__(self, 
            absolute_thres=0.5,
            absolute_start_epoch=20,
            relative_thres=0.2,
            key_to_watch='loss',
            *args, **kwargs):
        self.absolute_thres = absolute_thres
        self.absolute_start_epoch = absolute_start_epoch
        self.relative_thres = relative_thres
        self.key_to_watch = key_to_watch
        self.last_loss = None
        super().__init__(*args, **kwargs)

    def _run_validate(self, runner):
        runner.model.eval()

        results = self._get_valid_results(runner)
        agg_res = self.agg_func(results, **self.agg_func_kwargs)
        if runner.rank == 0:
            runner.logger.info({self.name: agg_res})

        value_to_use = agg_res[self.key_to_watch]
        if self.absolute_thres is not None \
                and runner.epoch >= self.absolute_start_epoch:
            if value_to_use > self.absolute_thres:
                sys.exit()
        if self.relative_thres is not None \
                and self.last_loss is not None:
            if value_to_use > self.last_loss + self.relative_thres:
                sys.exit()
        self.last_loss = value_to_use

        if runner.rank == 0:
            runner.record_saver.save(
                    {'validation_results': {self.name: agg_res}})

        runner.model.train()
