from Lane import Lane
import cv2
lanes = [Lane([679, 274, 606, 38], 700, 552, 'E', 'M'), Lane([679, 313, 606, 31], 850, 552, 'E', 'M'),
         Lane([679, 345, 606, 29], 1000, 552, 'E', 'S'), Lane([0, 410, 450, 31], 846, 642, 'W', 'M'),
         Lane([0, 386, 450, 29], 846, 642, 'W', 'M'), Lane([0, 352, 450, 27], 846, 642, 'W', 'N'),
         Lane([607, 500, 46, 230], 846, 642, 'S', 'M'), Lane([580, 500, 24, 230], 846, 642, 'S', 'M'),
         Lane([552, 500, 25, 230], 846, 642, 'S', 'W'), Lane([478, 0, 36, 246], 846, 642, 'N', 'M'),
         Lane([519, 0, 24, 246], 846, 642, 'N', 'M'), Lane([544, 0, 27, 246], 846, 642, 'N', 'E')]
opposite_lanes = [Lane([0, 295, 439, 56], 846, 642, 'W', 'W'), Lane([678, 380, 602, 71], 846, 642, 'E', 'E'),
                  Lane([567, 0, 61, 235], 846, 642, 'N', 'N'), Lane([485, 507, 61, 213], 846, 642, 'S', 'S'),Lane([436, 227, 242, 288], 846, 642, 'intersection', 'intersection'),
                  Lane([679, 274, 606, 38], 700, 552, 'E', 'M'), Lane([679, 313, 606, 31], 850, 552, 'E', 'M'),
                  Lane([679, 345, 606, 29], 1000, 552, 'E', 'S'), Lane([0, 410, 450, 31], 846, 642, 'W', 'M'),
                  Lane([0, 386, 450, 29], 846, 642, 'W', 'M'), Lane([0, 352, 450, 27], 846, 642, 'W', 'N'),
                  Lane([607, 500, 46, 230], 846, 642, 'S', 'M'), Lane([580, 500, 24, 230], 846, 642, 'S', 'M'),
                  Lane([552, 500, 25, 230], 846, 642, 'S', 'W'), Lane([478, 0, 36, 246], 846, 642, 'N', 'M'),
                  Lane([519, 0, 24, 246], 846, 642, 'N', 'M'), Lane([544, 0, 27, 246], 846, 642, 'N', 'E')
                  ]
def write_counts( counts, img, x, y, e, o):
     cv2.putText(img, f'{e}{o}:{counts}', (x, y), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
