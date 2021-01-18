import io
import base64
import math

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr

def img_b64_to_arr_basic(img_b64):
    """ Basic version of base64 string to numpy array.
    """
    img_decoded = base64.decodebytes(img_b64)
    img_array = np.frombuffer(img_decoded, dtype=np.int16)
    return img_array

def img_b64_to_pil(img_b64, num_channel=1):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    if num_channel == 1:
        return PIL.Image.open(f).convert('L')
    elif num_channel == 3:
        return PIL.Image.open(f).convert('RGB')

def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value, type='class'):
    """ Convert shapes to a mask with pixel labels
    """
    assert type in ['class', 'instance']
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    for shape in shapes:
        if shape['label']=='dummy':
            continue
        points = shape['points']
        label = shape['label']
        shape_type = shape.get('shape_type', None)
        if type == 'class':
            cls_name = label
        cls_id = label_name_to_value[cls_name]
        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
    return cls