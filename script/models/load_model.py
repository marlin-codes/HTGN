from script.utils.util import logger
from script.models.EvolveGCN.EGCN import EvolveGCN
from script.models.DynModels import DGCN
from script.models.HTGN import HTGN
from script.models.static_baselines import VGAENet, GCNNet


def load_model(args):
    if args.model in ['GRUGCN', 'DynGCN']:
        model = DGCN(args)
    elif args.model == 'HTGN':
        model = HTGN(args)
    elif args.model == 'EGCN':
        model = EvolveGCN(args)
    elif args.model == 'GAE':
        model = GCNNet()
    elif args.model == 'VGAE':
        model = VGAENet()
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(args.model))
    return model
