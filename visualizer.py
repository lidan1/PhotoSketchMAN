import numpy as np
import os
import ntpath
import time

class Visualizer():
    def __init__(self, id, name, ncols):
        # self.opt = opt
        self.display_id = id
        self.name = name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = 8097)
            self.display_single_pane_ncols = ncols

    def plot_current_errors(self, epoch, errors, winid):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch )
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'iter',
                'ylabel': 'loss'},
            win=winid)

# |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, winid):
        if winid > 0: # show images in the browser
                ncols = self.display_single_pane_ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                while idx % ncols != 0:
                    white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                self.vis.images(images, nrow=ncols, win=winid + 1,
                              opts=dict(title=title + ' images')) # pane col = image row
                # label_html = '<table style="border-collapse:separate;border-spacing:10px;">%s</table' % label_html
                # self.vis.text(label_html, win = winid + 2,
                #               opts=dict(title=title + ' labels'))
            # else:
            #     idx = 1
            #     for label, image_numpy in visuals.items():
            #         #image_numpy = np.flipud(image_numpy)
            #         self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
            #                            win=self.display_id + idx)
            #         idx += 1
