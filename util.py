"""
abbreviated util
"""

def get_frame_range(frame_selection):
    """ return array of selected frames"""
    if len(frame_selection) == 1:
        frame_range = frame_selection
    elif len(frame_selection) == 2:
        start = frame_selection[0]; end = frame_selection[1]
        frame_range = range(start, end + 1)
    elif len(frame_selection) == 3:
        start = frame_selection[0]; end = frame_selection[1]; rate = frame_selection[2]
        frame_range = range(start, end + 1, rate)
    else:
        print "Error: Must supply 1, 2, or 3 frame arguments\nWith one argument, plots single frame\nWith two arguments, plots range(start, end + 1)\nWith three arguments,$
        exit()

    return frame_range
