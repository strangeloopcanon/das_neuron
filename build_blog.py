"""Build a two-page essay and source-backed vector figures. Never refits a model."""
from pathlib import Path
import base64
import hashlib
import html
import json
import re

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Flowable
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle, Polygon
from reportlab.graphics import renderPDF, renderSVG

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/blog'
PDF = ROOT / 'output/pdf/neuron-labels.pdf'
INK = '#152D38'
MUTED = '#536974'
GRID = '#DCE4E7'
TEAL = '#087F78'
RUST = '#B24C30'
PAPER = '#FFFFFF'

TITLE = 'A neuron label is not a circuit explanation'
DECK = 'A tiny economic network shows what naming the parts leaves out.'

# These are the publication text, shared verbatim by the HTML and PDF editions.
PAGE_ONE = [
    ('p', 'I wanted to try a slightly silly experiment: take an economic model, turn it into a tiny neural network, and see what we understood. Could we name the neurons, inspect every weight, and still miss how the parts worked together?'),
    ('p', 'We built the experiment around a dynamic stochastic general equilibrium model, or DSGE. Households choose consumption, work and investment. Productivity changes at random. So does the discomfort of working. Installing capital has a cost. Decisions today affect what the household can do tomorrow.'),
    ('p', 'We solved the nonlinear economic equations numerically, then trained networks to approximate their decision rules. The inputs are capital, productivity and work discomfort. The example below has two layers of 16 hidden neurons and three outputs: consumption, labour and investment. That is 387 adjustable weights and biases. Every one is available to inspect.'),
    ('p', 'The distinction matters: the network approximates the solved decision rules at one calibration. It is not an exact translation of the entire economic model.'),
    ('p', 'The economic equations pose a similar problem. Knowing what capital and consumption mean does not solve the household\'s choices. Future returns, work effort and installation costs must fit together.'),
    ('h', 'What a neuron label tells us'),
    ('p', 'On 2,048 separate discovery examples, neuron H2.0 has a +0.941 correlation with capital. H2.4 tracks capital too (+0.806), but also productivity (+0.481). These are measured associations. Calling H2.0 the capital neuron would imply more than we have established.'),
    ('p', 'Hold capital fixed near its steady-state reference. Raise productivity and the discomfort of working together. H2.0 still contributes +0.01949 to the network\'s consumption change. Its capital label does not predict that contribution. Other neurons reinforce or offset it.'),
    ('fig', 'contributions'),
    ('p', 'For each final-layer neuron, multiply its activation change by its output weight and the output scale. Adding all 16 terms recovers the response to floating-point precision. Labels omit those signed contributions and cancellations.'),
]

PAGE_TWO = [
    ('h', 'The same labels, a different computation'),
    ('p', 'Now apply the work-discomfort shock alone, at low and high capital. The five largest consumption contributions share just one neuron. The weights and association labels stay fixed. The activation pattern and contribution ranking change with the state.'),
    ('fig', 'contributors'),
    ('p', 'Each hidden neuron adds a bias to a weighted sum, then sets a negative result to zero. A change can therefore switch neurons on or off downstream. A useful explanation must account for those switches and the resulting output. Naming what each unit tends to track does not do that.'),
    ('p', 'Jacob Hilton and colleagues\' <a href="https://distill.pub/2020/understanding-rl-vision/">Understanding RL Vision</a> goes further in another setting: it tests feature interpretations through targeted edits to a game-playing network. Our economic example illustrates why descriptions of features need a matching account of their joint effects.'),
    ('h', 'From equations to data to a circuit'),
    ('p', 'The training data come from the economic solution. For each state, we calculate consumption, labour and investment, then train the network on those input-output pairs. We assess its predictions on separate states. The economic equations supply a reference answer for the behaviour we want to explain.'),
    ('p', 'That gives us two ways to examine the same decision. The economic model describes household choices and their constraints. The neural circuit implements an approximation through weighted connections and activation switches. Inspecting either set of components still leaves the task of explaining how they combine.'),
    ('h', 'Six features in two coordinates'),
    ('p', 'We also built an exact, constructed circuit for six consumption-response features from six separate growth economies. Each feature has its own direction in a two-dimensional hidden representation. All six share two coordinates.'),
    ('p', 'The decoder recovers every feature exactly when one sector has the specified unit response. Opposite directions cancel when both sectors change, so the decoder misses both responses. This is designed superposition: more known features than dimensions, with measurable interference. The interactive version exposes the entire circuit.'),
    ('p', 'Knowing the weights lets us calculate each result. Naming the components does not supply that calculation. I want explanations that connect the parts to the decision, predict how they combine, and tell us what changes when we alter them.'),
]


def label(d, x, y, text, size=9, color=INK, anchor='start', font='Helvetica'):
    d.add(String(x, y, text, fontSize=size, fillColor=colors.HexColor(color),
                 fontName=font, textAnchor=anchor))


def contribution_figure(model, example):
    d = Drawing(480, 160)
    d.add(Rect(0, 0, 480, 160, fillColor=colors.white, strokeColor=None))
    label(d, 0, 148, 'Consumption change, decomposed by final hidden neuron', 10, font='Helvetica-Bold')
    terms = [row[0] for row in example['contrasts']['joint']['contribution']]
    assert abs(sum(terms) - example['contrasts']['joint']['delta'][0]) < 1e-12
    x0, x1, bottom, top = 43, 475, 33, 126
    ymin, ymax = -.025, .05
    sy = lambda v: bottom + (v-ymin)/(ymax-ymin)*(top-bottom)
    for v in [-.02, 0, .02, .04]:
        y = sy(v)
        d.add(Line(x0, y, x1, y, strokeColor=colors.HexColor(GRID), strokeWidth=.5))
        label(d, x0-6, y-3, f'{v:+.2f}' if v else '0', 8, MUTED, 'end')
    step = (x1-x0)/16
    for j, v in enumerate(terms):
        x = x0+(j+.5)*step
        if v:
            d.add(Rect(x-7, min(sy(0), sy(v)), 14, abs(sy(v)-sy(0)),
                       fillColor=colors.HexColor(TEAL if v>0 else RUST), strokeColor=None))
        else:
            d.add(Circle(x, sy(0), 1.6, fillColor=colors.HexColor(MUTED), strokeColor=None))
        label(d, x, 19, str(j), 8, MUTED, 'middle')
    label(d, 480, 4, 'Neuron index (H2); vertical units: model consumption', 8, MUTED, 'end')
    return d


def contributor_figure(model):
    rows = model['top5_route_overlap_low_high']['cost'][0]
    low, high = rows['low_units'], rows['high_units']
    common = set(low) & set(high)
    assert len(common) == 1 and rows['intersection_count'] == 1
    d = Drawing(480, 140)
    d.add(Rect(0, 0, 480, 140, fillColor=colors.white, strokeColor=None))
    label(d, 0, 128, 'Five largest consumption contributors to the work-discomfort shock', 10, font='Helvetica-Bold')
    label(d, 112, 108, 'Low capital: 0.5997 x reference', 9, MUTED, 'middle')
    label(d, 370, 108, 'High capital: 1.6007 x reference', 9, MUTED, 'middle')
    ys = [86-17*i for i in range(5)]
    for unit in common:
        d.add(Line(150, ys[low.index(unit)]+2, 310, ys[high.index(unit)]+2,
                   strokeColor=colors.HexColor(TEAL), strokeWidth=1.4))
    for x, units in [(112,low),(370,high)]:
        for i,u in enumerate(units):
            label(d, x-43, ys[i], f'{i+1}', 8, MUTED, 'end')
            label(d, x, ys[i], f'H2.{u}', 10, TEAL if u in common else INK,
                  'middle', 'Helvetica-Bold' if u in common else 'Helvetica')
    label(d, 240, 1, 'One shared neuron: H2.10. Ranked by absolute signed readout contribution.', 8, MUTED, 'middle')
    return d


CAPTIONS = {
    'contributions': 'Figure 1. Default network (H16, seed 113), capital = 0.9983 times its steady-state reference. Productivity and work discomfort rise together. Positive terms total +0.10048; negative terms total -0.06014; net change is +0.04034. These are model units, not percentage changes.',
    'contributors': 'Figure 2. Same default network, same work-discomfort shock, two fixed capital states. The line connects the only shared top-five unit. Each term includes that neuron\'s upstream computation; these are not individual input-to-output paths.',
}

METHODS = ('Methods: TRAIN 8,192; VALIDATION 2,048; DISCOVERY 2,048; TEST 8,192; '
           'INTERVENTION TEST 1,024. Samples are disjoint. Labels use DISCOVERY; shock figures use a separate deterministic grid. '
           'The results concern component labels and state-dependent computation. Full accuracy, interaction, and width diagnostics remain in the accompanying research reports. Economic checks establish a numerical first-order solution, not global optimality.')


class Graphic(Flowable):
    def __init__(self, drawing):
        super().__init__()
        self.drawing = drawing
        self.width, self.height = drawing.width, drawing.height

    def draw(self):
        renderPDF.draw(self.drawing, self.canv, 0, 0)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT/'system_analysis.json').read_text())
    richer = json.loads((ROOT/'richer_results.json').read_text())
    model = next(m for m in data['models'] if m['id'] == data['default_model_id'])
    example = model['examples'][1]
    assert model['id'] == 'H16-seed113'
    assert sum(len(row) for k,v in model['network'].items() for row in (v if k.startswith('W') else [v])) == 387
    assert abs(model['labels']['H2'][0]['correlations'][0] - .9414582287779749) < 1e-12
    assert example['interaction']['network'][0] > 0
    assert example['interaction']['teacher'][0] < 0
    # Frozen-gate removal prediction, 100% attenuation; report all three paired seeds.
    scale = {}
    for m in richer['models']:
        removal = next(t for t in m['interventions']['first_layer'] if t['fraction']==1)
        scale[(m['seed'],m['width'])] = removal['normalized_response_rmse']
    assert all(scale[(s,64)] < scale[(s,16)] for s in [113,229,347])
    figures = {'contributions':contribution_figure(model, example), 'contributors':contributor_figure(model)}
    for key,d in figures.items():
        renderSVG.drawToFile(d, str(OUT/f'{key}.svg'))
        renderPDF.drawToFile(d, str(OUT/f'{key}.pdf'))

    body = ParagraphStyle('body', fontName='Helvetica', fontSize=10, leading=13.5,
                          textColor=colors.HexColor(INK), spaceAfter=7)
    heading = ParagraphStyle('heading', parent=body, fontName='Helvetica-Bold', fontSize=12,
                             leading=15, spaceBefore=5, spaceAfter=8)
    title = ParagraphStyle('title', parent=body, fontName='Helvetica-Bold', fontSize=25,
                           leading=28, spaceAfter=7)
    deck = ParagraphStyle('deck', parent=body, fontSize=11, leading=15,
                          textColor=colors.HexColor(MUTED), spaceAfter=13)
    caption = ParagraphStyle('caption', parent=body, fontSize=8, leading=10.3,
                             textColor=colors.HexColor(MUTED), spaceAfter=8)
    methods = ParagraphStyle('methods', parent=caption, fontSize=7.4, leading=9.3, spaceBefore=3)
    story = [Paragraph(TITLE,title),Paragraph(DECK,deck)]
    for page_index, blocks in enumerate([PAGE_ONE, PAGE_TWO]):
        if page_index:
            story.append(PageBreak())
        for kind,text in blocks:
            if kind=='fig':
                story += [Graphic(figures[text]),Spacer(1,5),Paragraph(CAPTIONS[text],caption)]
            else:
                story.append(Paragraph(text,heading if kind=='h' else body))
    story.append(Paragraph(METHODS,methods))

    def footer(c, doc):
        c.setStrokeColor(colors.HexColor(GRID));c.setLineWidth(.5)
        c.line(66,38,546,38)
        c.setFillColor(colors.HexColor(MUTED));c.setFont('Helvetica',8)
        c.drawString(66,25,'A NEURON LABEL IS NOT A CIRCUIT EXPLANATION')
        c.drawRightString(546,25,str(doc.page))

    doc = SimpleDocTemplate(str(PDF), pagesize=(612,792), rightMargin=66,leftMargin=66,
                            topMargin=43,bottomMargin=49,title=TITLE,author='',
                            subject='A nonlinear economic network and the limits of neuron association labels')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

    css = '''
    :root{color-scheme:light} *{box-sizing:border-box}
    body{margin:0;background:#f4f5f2;color:#152d38;font:18px/1.6 Georgia,serif}
    article{max-width:820px;margin:40px auto;padding:50px 64px;background:white}
    h1{font:700 clamp(28px,5vw,42px)/1.1 system-ui,sans-serif;letter-spacing:-.035em;margin:0 0 12px}
    .deck{font:20px/1.4 system-ui,sans-serif;color:#536974;margin:0 0 30px}
    p{margin:0 0 18px}h2{font:700 23px/1.3 system-ui,sans-serif;margin:28px 0 16px}
    a{color:#087f78;text-decoration-thickness:1px;text-underline-offset:3px}
    figure{margin:26px 0}img{width:100%;height:auto;display:block}
    figcaption,.methods{font:13px/1.5 system-ui,sans-serif;color:#536974;margin-top:10px}
    .methods{border-top:1px solid #dce4e7;padding-top:16px;margin-top:26px}
    .demo{font:14px/1.5 system-ui,sans-serif;margin-top:18px}
    @media(max-width:650px){body{font-size:17px}article{padding:30px 22px;margin:0}}
    @media print{body{background:white}article{margin:0;max-width:none;padding:0}}
    '''
    fragments = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
                 f'<title>{html.escape(TITLE)}</title><style>{css}</style><article>',
                 f'<h1>{TITLE}</h1><p class="deck">{DECK}</p>']
    for blocks in [PAGE_ONE,PAGE_TWO]:
        for kind,text in blocks:
            if kind=='fig':
                svg = base64.b64encode((OUT/f'{text}.svg').read_bytes()).decode('ascii')
                fragments.append(f'<figure><img src="data:image/svg+xml;base64,{svg}" alt="{html.escape(CAPTIONS[text],quote=True)}"><figcaption>{html.escape(CAPTIONS[text])}</figcaption></figure>')
            else:
                tag='h2' if kind=='h' else 'p'
                fragments.append(f'<{tag}>{text}</{tag}>')
    fragments += [f'<p class="methods">{METHODS}</p>',
                  '<p class="demo"><a href="../../system.html">Explore the interactive policy circuit</a> · <a href="../../response_superposition.html">Six features in two coordinates</a> · <a href="../../richer_report.md">Experimental results and methods</a></p>', '</article></html>']
    (OUT/'neuron-labels.html').write_text('\n'.join(fragments))
    evidence = dict(model_id=model['id'], capital_over_reference=example['capital_over_kss'],
        joint_consumption_delta=example['contrasts']['joint']['delta'][0],
        positive_contributions=example['contrasts']['joint']['positive_sum'][0],
        negative_contributions=example['contrasts']['joint']['negative_sum'][0],
        unit0_contribution=example['contrasts']['joint']['contribution'][0][0],
        top5_cost_contributors=model['top5_route_overlap_low_high']['cost'][0],
        scaling_removal_prediction_error={f'{s}/{w}':v for (s,w),v in scale.items()},
        source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['system_analysis.json','richer_results.json']})
    (OUT/'figure-data.json').write_text(json.dumps(evidence,indent=2)+'\n')
    text = ' '.join(re.sub('<[^>]+>','',s) for page in [PAGE_ONE,PAGE_TWO] for k,s in page if k!='fig')
    print(json.dumps(dict(words=len(text.split()),pdf=str(PDF),html=str(OUT/'neuron-labels.html'))))


if __name__=='__main__':
    build()
