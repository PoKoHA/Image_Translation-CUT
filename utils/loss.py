from packaging import version
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):

    def __init__(self, args):
        super(PatchNCELoss, self).__init__()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patchees = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # positive logit
        l_pos = torch.bmm(feat_q.view(num_patchees, 1, -1), feat_k.view(num_patchees, -1, 1))
        l_pos = l_pos.view(num_patchees, 1)

        # negative logit
        # 논문) 외부 Image들로부터 Negative Patch를 구하는 것보다 내부 이미지에서 구하는 것이 더 좋은 효과를 줌.
        # 하지만 Single Image Translation 같은 경우 batch들이 'Same Resolution'에서 crop한 것들로 구성되어 있으므로
        # 이런 경우 모든 minibatch를 활용홤.
        if self.args.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.args.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        # exp(-10)적용 / cross entropy에 Exp 포함되어 있음
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.args.nce_T # default: 0.07

        loss = self.cross_entropy(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

        return loss


def calculate_NCE_loss(args, src, tgt, generator, patchMLP, nce_layers, flipped_for_equivariance,
                       criterion_NCE):

    n_layers = len(args.nce_layers)
    feat_q = generator(tgt, nce_layers, encode_only=True)

    if args.flip_equivariance and flipped_for_equivariance:
        feat_q = [torch.flip(fq, [3]) for fq in feat_q]

    feat_k = generator(src, nce_layers, encode_only=True)
    feat_k_pool, sample_ids = patchMLP(feat_k, args.num_patches, None)
    feat_q_pool, _ = patchMLP(feat_q, args.num_patches, sample_ids)

    total_nce_loss = 0.0
    for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, criterion_NCE, nce_layers):
        loss = crit(f_q, f_k) * args.lambda_NCE
        total_nce_loss += loss.mean()

    return total_nce_loss / n_layers


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss