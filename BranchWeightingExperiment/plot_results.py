"""Replot stored numerical results; imports no training framework."""
from pathlib import Path
TEXT_COLOR = '#263238'


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    for side in ('left','bottom'):
        axis.spines[side].set_color('#A5ADB3')
        axis.spines[side].set_linewidth(.65)
    axis.tick_params(axis='both',colors='#59636B',labelsize=7.7,
                     length=2.5,width=.6,pad=2.)


def render(payload, output):
    """Presentation only: values, sampling points and axis limits remain unchanged."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, NullLocator
    updates=payload['updates']
    initials=payload['protocol']['initial_probabilities']
    ks=payload['protocol']['K']
    results=payload['initializations']
    plt.rcParams.update({
        'font.family':'serif','font.serif':['DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':9.5,
        'pdf.fonttype':42,'ps.fonttype':42,
        'text.color':TEXT_COLOR,'axes.labelcolor':TEXT_COLOR,
        'axes.titlepad':6,'lines.solid_capstyle':'round',
        'lines.dash_capstyle':'butt','savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,5,figsize=(8.2,3.65),sharex=True,sharey=True)
    fig.subplots_adjust(left=.125,right=.985,bottom=.16,top=.825,wspace=.19,hspace=.20)
    # Order and names follow the user's legend; methods retain their original colors.
    styles=[('naive','Naive','#74828D',(0,(4.0,2.0)),1.55),
            ('treerl_style_sqrt','Heuristic (TreeRL-style)','#277DA1',(0,(5.0,1.7,1.0,1.7)),1.55),
            ('weighted','Branch-weighted (TTPG)','#B5475D','-',1.85)]
    for row,p0 in enumerate(initials):
        for col,k in enumerate(ks):
            ax=axes[row,col]
            style_axis(ax)
            for order,(method,label,color,style,width) in enumerate(styles):
                ax.plot(updates,results[str(p0)]['curves'][method][str(k)],
                        color=color,ls=style,lw=width,label=label,zorder=3+order)
            if row==0:ax.set_title(f'$K={k}$',fontsize=10.5,weight='regular',pad=7)
            ax.set_xscale('symlog',linthresh=10)
            ax.set_xlim(0,payload['protocol']['updates_count']);ax.set_ylim(0,.265)
            ax.set_xticks([0,100,10000]);ax.set_xticklabels(['0',r'$10^2$',r'$10^4$'])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_yticks([0,.10,.20,.25]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='both',labelsize=9,length=2.5,width=.65,pad=2.8)
            ax.grid(axis='y',color='#D7DDE2',linestyle=(0,(2.5,3)),lw=.5,alpha=.60)
            ax.spines['left'].set_color('#A8B0B7');ax.spines['bottom'].set_color('#A8B0B7')
        box=axes[row,0].get_position()
        fig.text(.065,(box.y0+box.y1)/2,f'$p_0={p0:g}$',
                 rotation=90,ha='center',va='center',fontsize=10.5,color=TEXT_COLOR)
    handles=[Line2D([],[],label=label,color=color,ls=style,lw=width)
             for _,label,color,style,width in styles]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.955),
               ncol=3,frameon=False,fontsize=9.5,handlelength=2.8,
               columnspacing=1.5,handletextpad=.65,borderaxespad=.4)
    fig.supxlabel('Policy gradient updates (symlog scale)',fontsize=10.5,y=.060)
    fig.supylabel('Exact expected return',fontsize=10.5,x=.035)
    stem=output/'exact_branch_weighting_initializations'
    for ext in ('png','pdf'):
        fig.savefig(stem.with_suffix('.'+ext),dpi=300,bbox_inches='tight',pad_inches=.045)
    plt.close(fig)
