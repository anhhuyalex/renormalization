import torch
import torch.nn as nn
# from .dg import DG
from openselfsup.models import builder
# from cls_module.components.label_learner import LabelLearner


class STM_AHA(nn.Module):

    global_key = 'memory'
    local_key = None

    def __init__(self, input_size, dg_cfg,
                 perforant_cfg, ca3_cfg, msp_cfg,
                 is_hebbian_perforant = True, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.dg_cfg = dg_cfg
        self.perforant_cfg = perforant_cfg
        self.ca3_cfg = ca3_cfg
        self.msp_cfg = msp_cfg
        self.is_hebbian_perforant = is_hebbian_perforant
        self.build()

        self.step = 0
        self.initial_state = self.state_dict()

#         if self.global_key in self.config:
#             self.config = self.config[self.global_key]

    def build(self):
        """Build AHA as short-term memory module."""

        # Build the Dentae Gyrus
        self.dg_mlp = builder.build_neck(self.dg_cfg)
        # DG(self.input_shape, self.config['dg']).to(self.device)
        # dg_output_shape = [1, self.config['dg']['num_units']]

        # Build the Perforant Pathway
        self.perforant = builder.build_neck(self.perforant_cfg)
        # PerforantHebb(ec_shape=self.input_shape,
        #                                       dg_shape=dg_output_shape,
        #                                       ca3_shape=dg_output_shape,
        #                                       config=self.config['perforant_hebb'])
        # else:
        #     self.perforant = PerforantPR(self.input_shape, dg_output_shape, self.config['perforant_pr'])

        # Build the CA3
        self.ca3 = builder.build_neck(self.ca3_cfg)
        # KNNBuffer(input_shape=dg_output_shape, target_shape=dg_output_shape, config=self.config['ca3'])

        # Build the Monosynaptic Pathway
        self.msp = builder.build_neck(self.msp_cfg)
        
#         # Optionally between a bioligically plausible MSP (CA1, CA3 => CA1 pathways) or a simple pattern mapper
#         if self.config.get('msp_type', None) == 'ca1':
#             self.msp = MonosynapticPathway(ca3_shape=ca3_output_shape, ec_shape=self.input_shape, config=self.config['msp'])
#         else:
#             self.pm = PatternMapper(ca3_output_shape, self.target_shape, self.config['pm'])
#             self.pm_ec = PatternMapper(ca3_output_shape, self.input_shape, self.config['pm_ec'])

#         # Build the Label Learner module
#         if 'classifier' in self.config:
#             self.build_classifier(input_shape=ca3_output_shape)

#         self.output_shape = ca3_output_shape
    def compute_output_shape(self, fn, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape

        with torch.no_grad():
            sample_output = fn(torch.rand(1, *(input_shape[1:])))
            output_shape = list(sample_output.data.shape)
            output_shape[0] = -1
        return output_shape

    def add_optimizer(self, name, optimizer):
        setattr(self, name + '_optimizer', optimizer)

    def build_classifier(self, input_shape):
        """Optionally build a classifier."""
        if not 'classifier' in self.config:
            raise KeyError('Classifier configuration not found.')

        classifier = LabelLearner(input_shape, self.config['classifier'])
        self.add_module('classifier', classifier)

    def forward_one_time(self, inputs):
        """
        seq_vecs is a sequence of embeddings of shape (seq_to_memorize+1, input_shape)
            The first seq_to_memorize elements contain the sequence to memorize
            The last element contains the noisy cue
        """

        losses = {}
        outputs = {}
        features = {}
        

        # DG
        # DG is not trainable and only has 1 mode
        dg_out = self.dg_mlp(inputs) 
        outputs['dg'] = dg_out
        # print("dg_out", dg_out.shape)
        # features['dg'] = outputs['dg'].detach().cpu()

        # Compute DG Overlap
        # overlap = self.dg.compute_overlap(outputs['dg'])
        # losses['dg_overlap'] = overlap.sum()

        # Perforant Pathway: Hebbian Learning
        if self.is_hebbian_perforant: 
            ca3_cue, losses['dg_ca3'], losses['ec_ca3'], losses['ca3_cue'] = self.perforant(ec_inputs=inputs, dg_inputs=dg_out , mode="study")

        # Perforant Pathway: Error-Driven Learning
        # Not yet implemented
        else:
            pr_targets = outputs['dg']
            # pr_targets = outputs['dg'] if self.training else self.pc_buffer_batch
            losses['pr'], outputs['pr'] = self.perforant(inputs=inputs, targets=pr_targets)
            features['pr'] = outputs['pr']['pr_out'].detach().cpu()

            # Compute PR Mismatch
            pr_out = outputs['pr']['pr_out']
            pr_batch_size = pr_out.shape[0]
            losses['pr_mismatch'] = torch.sum(torch.abs(pr_targets - pr_out)) / pr_batch_size

            ca3_cue = outputs['dg'] if self.training else outputs['pr']['z_cue']



            # print("ca3_cue", ca3_cue.shape)

        # CA3
        ca3_output = self.ca3(inputs=ca3_cue, mode="study")
        outputs['ca3'] = ca3_output 
        # features['ca3'] = outputs['ca3'].detach().cpu()

        # outputs['encoding'] = outputs['ca3'].detach()
        # outputs['output'] = outputs['ca3'].detach()

        # Monosynaptic Pathway
        if self.msp_cfg.get('type', None) == 'MonosynapticPathway_AHA':
            # During study, EC will drive the CA1
            losses['ca1'], ca1_outputs, losses['ca3_ca1'], ca3_ca1_outputs = self.msp(ec_inputs=inputs,
                                                                                        ca3_inputs=ca3_output, mode="study")
            outputs['ca1'] = ca1_outputs
            outputs['ca3_ca1'] = ca3_ca1_outputs
            outputs['decoding'] = None
            features['recon'] = None

            outputs['decoding_ec'] = ca1_outputs['decoding'].detach()
            features['recon_ec'] =  ca1_outputs['decoding'].detach().cpu()
        else:
            losses['pm'], outputs['pm'] = self.pm(inputs=outputs['ca3'], targets=targets)
            losses['pm_ec'], outputs['pm_ec'] = self.pm_ec(inputs=outputs['ca3'], targets=inputs)

            outputs['decoding'] = outputs['pm']['decoding'].detach()
            features['recon'] = outputs['pm']['decoding'].detach().cpu()

            outputs['decoding_ec'] = outputs['pm_ec']['decoding'].detach()
            features['recon_ec'] =  outputs['pm_ec']['decoding'].detach().cpu()

        self.features = features
        
        
        return losses, outputs
    
    def recall(self, cue):
        # DG
        dg_out = self.dg_mlp(cue) 
        
        # Perforant Pathway: Hebbian Learning
        if self.is_hebbian_perforant: 
            ca3_cue, _, _, _ = self.perforant(ec_inputs=cue, dg_inputs=dg_out , mode="recall")

        # CA3
        ca3_output = self.ca3(inputs=ca3_cue, mode="recall")
        
        if self.msp_cfg.get('type', None) == 'MonosynapticPathway_AHA':
            _, ca1_outputs, _, ca3_ca1_outputs = self.msp(ec_inputs=cue, ca3_inputs=ca3_output, mode="recall")
            
        retrieved_item = ca1_outputs["decoding"]
        # print("retrieved_item", retrieved_item.shape)
        return [retrieved_item]
    
    def set_study_mode(self, train_mode=True):
        self.perforant.train(train_mode)
        self.ca3.train(train_mode)
        self.msp.train(train_mode)
        
    def reset(self):
        """Reset modules and optimizers."""
        for _, module in self.named_children():
            if hasattr(module, 'reset'):
                module.reset()
                
    def forward(self, seq_vecs, mode="train"):  # pylint: disable=arguments-differ
        """Perform an optimisation step with the entire CLS module.
            seq_vecs is a sequence of vectors, with the first n-1 vectors
            to be elements to be recalled
            The final element of the sequence is the cue vector
        """
        losses = {}
        outputs = {}
        num_seq = seq_vecs.size(0)
        # print("seq_vecs", seq_vecs.shape)
        
        # In training mode, set self.perforant, self.ca3, and self.msp to training
        if mode=="train":
            self.set_study_mode(train_mode=True)
        # self.perforant.train(False)
        # Encoding embeddings
        self.ca3.reset()
        self.perforant.reset()
        # print("buffer", self.ca3.buffer)
        for idx in range(num_seq-1):
            memory_loss, memory_outputs = self.forward_one_time(inputs = seq_vecs[idx])
            # print("buffer mode", self.ca3.buffer.shape)
        # print("buffer ", self.ca3.buffer[:, 0, :].shape)
        # print("buffer ", self.ca3.buffer.permute(1, 0, 2)[0, :, :].shape)
        perf = list(self.perforant.parameters())
        self.set_study_mode(train_mode=False)
        retrieved_item = self.recall(cue = seq_vecs[-1])
        
        # raise ValueError
        # clear ca3, perforant buffer
        
        self.step += 1

        return retrieved_item