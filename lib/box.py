import numpy as np
import cv2, math

class Box:
  def __init__(this, rotation = 0, bounding_box = (0, 0, 0, 0)):
    this.rotation = rotation
    this.bounding_box = bounding_box

  def draw(this, image, colour = (0, 255, 0), offset = (0, 0)):
    image_h, image_w, _ = image.shape
    offset_x, offset_y = offset
    top_x, top_y, bottom_x, bottom_y = this.bounding_box
    angle = this.rotation

    start_x, start_y = int((offset_x + top_x) * image_w), int((offset_y + top_y) * image_h)
    end_x, end_y = int((offset_x + bottom_x) * image_w), int((offset_y + bottom_y) * image_h)

    # start_x = offset_x + math.cos(angle) * (start_x - offset_x) - math.sin(angle) * (start_y - offset_y)
    # start_y = offset_y + math.sin(angle) * (start_x - offset_x) + math.cos(angle) * (start_y - offset_y)
    
    # end_x = offset_x + math.cos(angle) * (end_x - offset_x) - math.sin(angle) * (end_y - offset_y)
    # end_y = offset_y + math.sin(angle) * (end_x - offset_x) + math.cos(angle) * (end_y - offset_y)

    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0,100,0), 1)


    pts = np.array([
      this.rotate(offset, (start_x, start_y), angle),
      this.rotate(offset, (start_x, end_y), angle),
      this.rotate(offset, (end_x, end_y), angle),
      this.rotate(offset, (end_x, start_y), angle)
      ], np.int32
    )
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image, [pts], True, colour)

  def rotate(this, origin, point, angle):
    ox, oy = origin
    px, py = point
    return [ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy), oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)]

  def set_rotation(this, rotation):
    this.rotation = rotation
    
  def set_bounding_box(this, bounding_box):
    this.bounding_box = bounding_box

  def get_rotation(this):
    return this.rotation

  def get_bounding_box(this):
    return this.bounding_box