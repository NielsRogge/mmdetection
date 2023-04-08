import argparse
from collections import OrderedDict

import numpy as np
import torch


def convert(src, dst=None):

    src_model = torch.load(src, map_location="cpu")
    print(src_model.keys())

    dst_state_dict = OrderedDict()
    for k, v in src_model['model'].items():
        key_name_split = k.split('.')
        if 'backbone.fpn_lateral' in k:
            lateral_id = int(key_name_split[-2][-1])
            name = f'neck.lateral_convs.{lateral_id-2}.conv.{key_name_split[-1]}'
        elif 'backbone.fpn_output' in k:
            lateral_id = int(key_name_split[-2][-1])
            name = f'neck.fpn_convs.{lateral_id-2}.conv.{key_name_split[-1]}'
        elif 'backbone.bottom_up.stem.conv1.norm.' in k:
            name = f'backbone.bn1.{key_name_split[-1]}'
        elif 'backbone.bottom_up.stem.conv1.' in k:
            name = f'backbone.conv1.{key_name_split[-1]}'
        elif 'backbone.bottom_up.backbone' in k:
            name = k.replace('backbone.bottom_up.backbone', 'backbone')
        elif 'proposal_generator.anchor_generator' in k:
            continue
        elif 'rpn' in k:
            if 'conv' in key_name_split[2]:
                name = f'rpn_head.rpn_conv.{key_name_split[-1]}'
            elif 'objectness_logits' in key_name_split[2]:
                name = f'rpn_head.rpn_cls.{key_name_split[-1]}'
            elif 'anchor_deltas' in key_name_split[2]:
                name = f'rpn_head.rpn_reg.{key_name_split[-1]}'
            else:
                print(f'{k} is invalid')
        elif 'roi_heads' in k:
            if key_name_split[1] == 'box_head':
                fc_id = int(key_name_split[2][-1]) - 1
                name = f'roi_head.bbox_head.shared_fcs.{fc_id}.{key_name_split[-1]}'
            elif 'cls_score' == key_name_split[2]:
                name = f'roi_head.bbox_head.fc_cls.{key_name_split[-1]}'
            elif 'bbox_pred' == key_name_split[2]:
                name = f'roi_head.bbox_head.fc_reg.{key_name_split[-1]}'
            elif 'mask_fcn' in key_name_split[2]:
                conv_id = int(key_name_split[2][-1]) - 1
                name = f'roi_head.mask_head.convs.{conv_id}.conv.{key_name_split[-1]}'
            elif 'deconv' in key_name_split[2]:
                name = f'roi_head.mask_head.upsample.{key_name_split[-1]}'
            elif 'roi_heads.mask_head.predictor' in k:
                name = f'roi_head.mask_head.conv_logits.{key_name_split[-1]}'
            elif 'roi_heads.mask_coarse_head.reduce_spatial_dim_conv' in k:
                name = f'roi_head.mask_head.downsample_conv.conv.{key_name_split[-1]}'
            elif 'roi_heads.mask_coarse_head.prediction' in k:
                name = f'roi_head.mask_head.fc_logits.{key_name_split[-1]}'
            elif key_name_split[1] == 'mask_coarse_head':
                fc_id = int(key_name_split[2][-1]) - 1
                name = f'roi_head.mask_head.fcs.{fc_id}.{key_name_split[-1]}'
            elif 'roi_heads.mask_point_head.predictor' in k:
                name = f'roi_head.point_head.fc_logits.{key_name_split[-1]}'
            elif key_name_split[1] == 'mask_point_head':
                fc_id = int(key_name_split[2][-1]) - 1
                name = f'roi_head.point_head.fcs.{fc_id}.conv.{key_name_split[-1]}'
            else:
                print(f'{k} is invalid')
        else:
            print(f'{k} is not converted!!')

        if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
            raise ValueError(
                'Unsupported type found in checkpoint! {}: {}'.format(
                    k, type(v)))
        elif isinstance(v, torch.Tensor):
            print("Name:", name)
            dst_state_dict[name] = v

    mmdet_model = dict(state_dict=dst_state_dict, meta=dict())
    # torch.save(mmdet_model, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    # parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src)


if __name__ == '__main__':
    main()