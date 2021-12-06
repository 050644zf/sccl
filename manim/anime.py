from typing_extensions import runtime
from debugpy.common.json import enum
from manimlib import *
import numpy as np
import matplotlib.pyplot as plt

class WriteStuff(Scene):
    def construct(self):
        example_text = Text(
            "这是文本",
            t2c={"text": YELLOW},
            font='Noto Sans CJK SC Bold'
        )
        example_tex = Tex(
            "\\sum_{k=1}^\\infty {1 \\over k^2} = {\\pi^2 \\over 6}",
        )
        group = VGroup(example_text, example_tex)
        group.arrange(DOWN)
        #group.set_width(FRAME_WIDTH - 2 *MED_SMALL_BUFF)

        self.play(Write(example_text))
        self.play(Write(example_tex))
        self.wait()

class pre(Scene):
    def construct(self):
        def normal(mu, sigma, n):
            return np.random.normal(mu,sigma,n).reshape([-1,1])

        def getp(vec, centers, k):
            _a = (1+np.linalg.norm(vec - centers[k])**2) ** -1
            _b = 0
            for _k in range(len(centers)):
                _b += (1+np.linalg.norm(vec - centers[_k])**2) ** -1

            return  _a / _b

        def getap(qmat, f_k, j, k):
            _a = qmat[j,k]**2 / f_k[k]
            _b = 0
            for idx,_f in enumerate(f_k):
                _b += qmat[j,idx]**2/f_k[idx]
            
            return _a/_b

        def getLoss(pmat, qmat, j):
            l = 0
            for k in range(pmat.shape[1]):
                l += pmat[j,k] * np.log(pmat[j,k]/qmat[j,k])

            return l

        def getCL(args):
            samples, c0, c1 = args
            centers = np.array([[c0],[c1]])
            qmat = np.zeros([len(samples),len(centers)])
            for j in range(qmat.shape[0]):
                for k in range(qmat.shape[1]):
                    qmat[j,k] = getp(samples[j], centers, k)

            f_k = qmat.sum(axis=0)

            pmat = np.zeros([len(samples),len(centers)])
            for j in range(pmat.shape[0]):
                for k in range(pmat.shape[1]):
                    pmat[j,k] = getap(qmat, f_k, j, k)

            loss = np.zeros([len(samples)])
            for j in range(loss.shape[0]):
                loss[j] = getLoss(pmat,qmat,j)

            totalLoss = loss.sum()/len(samples)
            return qmat, pmat, loss, totalLoss

        samples = np.concatenate((normal(0.0,0.5,100),normal(5.0,0.5,100)))
        samples.sort(axis=0)
        hbins, bins, _ = plt.hist(samples.reshape([-1]), bins=20)
        hbins /= np.max(hbins)

        class result():
            def __init__(self,samples, axes,axes2,c0=0,c1=5):
                self.samples = samples
                self.axes = axes
                self.axes2 = axes2
                self.qmat, self.pmat, self.loss, self.totalLoss = getCL((self.samples,c0,c1))
                self.qdots = VGroup()
                self.q2dots = VGroup()
                self.pdots = VGroup()
                self.ldots = VGroup()
                self.tdot = Dot()
                self.tdot.set_color(PURPLE_C)
                self.tdot.move_to(self.axes2.c2p(c1 - 5,self.totalLoss))
                for idx in range(len(samples)):
                    dot = SmallDot()
                    dot.set_color(YELLOW_D)
                    dot.axes = self.axes
                    dot.move_to(self.axes.c2p(self.samples[idx,0],self.qmat[idx,0]))
                    self.qdots.add(dot)

                    dot = SmallDot()
                    dot.set_color(TEAL_D)
                    dot.axes = self.axes
                    dot.move_to(self.axes.c2p(self.samples[idx,0],self.qmat[idx,1]))
                    self.q2dots.add(dot)

                    dot = SmallDot()
                    dot.set_color(BLUE_D)
                    dot.axes = self.axes
                    dot.move_to(self.axes.c2p(self.samples[idx,0],self.pmat[idx,0]))
                    self.pdots.add(dot)

                    dot = SmallDot()
                    dot.set_color(GREEN_D)
                    dot.axes = self.axes
                    dot.move_to(self.axes.c2p(self.samples[idx,0],self.loss[idx]))
                    self.ldots.add(dot)


            def update(self,c0,c1):
                self.qmat, self.pmat, self.loss, self.totalLoss = getCL((self.samples,c0,c1))
                actList = []
                for idx in range(len(self.samples)):
                    self.qdots[idx].generate_target()
                    self.qdots[idx].target.move_to(self.axes.c2p(self.samples[idx,0],self.qmat[idx,0]))
                    self.pdots[idx].generate_target()
                    self.pdots[idx].target.move_to(self.axes.c2p(self.samples[idx,0],self.pmat[idx,0]))
                    self.ldots[idx].generate_target()
                    self.ldots[idx].target.move_to(self.axes.c2p(self.samples[idx,0],self.loss[idx]))
                    #actList.append(self.qdots[idx].animate.move_to(self.axes.c2p(self.samples[idx,0],self.qmat[idx,0])))
                    #actList.append(self.pdots[idx].animate.move_to(self.axes.c2p(self.samples[idx,0],self.pmat[idx,0])))
                
                #return actList
                self.tdot.generate_target()
                self.tdot.target.move_to(self.axes2.c2p(c1 - 5,self.totalLoss))


            

        axes = Axes(
            x_range=(-2, 8,1),
            y_range=(-0.1, 1.2,10),
            # 坐标系将会伸缩来匹配指定的height和width
            height=6,
            width=10,
            # Axes由两个NumberLine组成，你可以通过axis_config来指定它们的样式
            x_axis_config={
                "stroke_color": GREY_A,
                "stroke_width": 1,
                "include_numbers": True,
                "numbers_to_exclude": []
            },
            y_axis_config={
                "include_tip": False,
                "stroke_width": 0,
                "include_ticks": False
            }
        )

        axes2 = Axes(
            x_range=(-3, 3,1),
            y_range=(-0.01, 0.1,0.02),
            # 坐标系将会伸缩来匹配指定的height和width
            height=6,
            width=10,
            # Axes由两个NumberLine组成，你可以通过axis_config来指定它们的样式
            x_axis_config={
                "stroke_color": GREY_A,
                "stroke_width": 1,
                "include_numbers": True
            }
        )

        #axes.add_background_rectangle(color=GREEN)        
        

        ag = VGroup(axes, axes2)
        ag.arrange()
        axes2.shift(DOWN*0.1+RIGHT*6)
        axes.center()

        self.play(ShowCreation(axes), run_time=1)
        #self.play(ShowCreation(axes2))


        abins = []
        for idx,hbin in enumerate(hbins):
            xpos = bins[idx] + bins[idx+1]
            xpos /= 2
            #print(xpos)
            abins.append(axes.get_v_line(axes.c2p(xpos,hbin),line_func=Line, color=DARK_BROWN, stroke_width=10))
        

        self.play(*[ShowCreation(i) for i in abins], run_time=3)

        c0 = ValueTracker(0)
        c1 = ValueTracker(5)

        c_0 = Tex("\hat{\mu_1}")
        c_1 = Tex("\hat{\mu_2}")
    
        q_j1 = Tex("q_{j1}",color=YELLOW_D)
        q_fml = Tex(r"=\frac{\left(1+\lvert e_{j}-\mu_{1}\rvert_{2}^{2} / \alpha\right)^{-\frac{\alpha+1}{2}}}{\sum_{k^{\prime}=1}^{K}\left(1+\lvert e_{j}-\mu_{k^{\prime}}\rvert_{2}^{2} / \alpha\right)^{-\frac{\alpha+1}{2}}}", color=YELLOW_D)
        #q_fml = SVGMobject(file_name="videos/q_fml.svg", color=YELLOW_D)

        q_j2 = Tex("q_{j2}",color=TEAL_D)
    
        p_j1 = Tex("p_{j1}",color=BLUE_D)
        p_fml = Tex(r"=\frac{q_{j k}^{2} / f_{1}}{\sum_{k^{\prime}} q_{j k}^{2} / f_{k^{\prime}}}",color=BLUE_D)
        #p_fml = SVGMobject(file_name="videos/p_fml.svg", color=BLUE_D)

        loss = Tex("\ell_{j}^{C}",color=GREEN_D)
        loss_fml = Tex(r"=\mathbf{K L}\left[p_{j} \vert q_{j}\right]=\sum_{k=1}^{K} p_{j k} \log \frac{p_{j k}}{q_{j k}}",color=GREEN_D)
        #loss_fml = SVGMobject(file_name="videos/loss_fml.svg", color=GREEN_D)

        Loss = Tex("\mathcal{L}",color=PURPLE_C)



        '''
        qdots0 = VGroup()
        qdots1 = VGroup()
        f_k0 = ValueTracker(0)
        f_k1 = ValueTracker(0)
        f_k0.add_updater(lambda m: m.set_value(axes.p2c([0,sum([d.get_center()[1] for d in qdots0]),0])[1]))
        f_k1.add_updater(lambda m: m.set_value(axes.p2c([0,sum([d.get_center()[1] for d in qdots1]),0])[1]))
        
        for sample in samples:
            dot = SmallDot()
            dot.set_color(YELLOW_C)
            dot.axes = axes
            dot.move_to(dot.axes.c2p(sample[0],0))
            dot.add_updater(lambda m: m.set_y(getpfromdots(m,c_0,c_1,0)))
            qdots0.add(dot)

            dot = SmallDot()
            dot.set_color(YELLOW_C)
            dot.axes = axes
            dot.move_to(dot.axes.c2p(sample[0],0))
            dot.add_updater(lambda m: m.set_y(getpfromdots(m,c_0,c_1,1)))
            qdots1.add(dot)
        '''
        c_0.generate_target()
        c_1.generate_target()
        c_0.add_updater(lambda m: m.target.move_to(axes.c2p(c0.get_value(),-0.2)))
        c_1.add_updater(lambda m: m.target.move_to(axes.c2p(c1.get_value(),-0.2)))

        c_0.move_to(c_0.target.get_center())
        c_1.move_to(c_1.target.get_center())

        r = result(samples,axes,axes2,c0.get_value(),c1.get_value())

        q_j1.add_updater(lambda m: m.move_to(r.qdots[0].get_center() + LEFT + DOWN*0.2))
        q_fml.add_updater(lambda m: m.next_to(q_j1.get_right()))
        q_j2.add_updater(lambda m: m.move_to(r.q2dots[0].get_center() + LEFT + DOWN*0.2))
        p_j1.add_updater(lambda m: m.move_to(r.pdots[0].get_center() + LEFT + UP*0.2))
        p_fml.add_updater(lambda m: m.next_to(p_j1.get_right() + 0.5*RIGHT))
        loss.add_updater(lambda m: m.move_to(r.ldots[0].get_center() + LEFT))
        loss_fml.add_updater(lambda m: m.next_to(loss.get_right() + 0.5*RIGHT))
        
        self.play(Write(c_0),Write(c_1))
        #print(f_k[0].get_value(),f_k[1].get_value())
        

        self.wait(1)

        #self.play(ShowCreation(r.qdots), DrawBorderThenFill(q_j1))
        self.play(DrawBorderThenFill(q_fml), DrawBorderThenFill(q_j1))
        self.wait(2)
        self.play(Uncreate(q_fml), DrawBorderThenFill(r.qdots))
        self.wait(1)
        self.play(DrawBorderThenFill(r.q2dots), DrawBorderThenFill(q_j2))
        self.wait(2)
        self.play(FadeOut(r.q2dots, DOWN), FadeOut(q_j2, DOWN))

        self.wait(1)


        #self.play(*[GrowFromPoint(r.pdots[i],r.qdots[i]) for i in range(len(samples))], DrawBorderThenFill(p_j1))
        self.play(DrawBorderThenFill(p_fml), DrawBorderThenFill(p_j1))
        self.wait(2)
        self.play(Uncreate(p_fml), DrawBorderThenFill(r.pdots))
        self.wait(1)
        #self.play(*[GrowFromPoint(r.ldots[i],r.pdots[i]) for i in range(len(samples))])

        t = 5

        while t>2:
            t-=0.5
            c1.set_value(t)
            r.update(c0.get_value(),c1.get_value())
            self.play(*[MoveToTarget(d) for d in r.qdots], *[MoveToTarget(d) for d in r.pdots],*[MoveToTarget(d) for d in [c_0,c_1]], run_time=0.2, rate_func=linear)


        while t<8:
            t+=0.5
            c1.set_value(t)
            r.update(c0.get_value(),c1.get_value())
            self.play(*[MoveToTarget(d) for d in r.qdots], *[MoveToTarget(d) for d in r.pdots],*[MoveToTarget(d) for d in [c_0,c_1]], run_time=0.2, rate_func=linear)

        while t>2:
            t-=0.5
            c1.set_value(t)
            r.update(c0.get_value(),c1.get_value())
            self.play(*[MoveToTarget(d) for d in r.qdots], *[MoveToTarget(d) for d in r.pdots],*[MoveToTarget(d) for d in [c_0,c_1]], run_time=0.2, rate_func=linear)


        while t<8:
            t+=0.5
            c1.set_value(t)
            r.update(c0.get_value(),c1.get_value())
            self.play(*[MoveToTarget(d) for d in r.qdots], *[MoveToTarget(d) for d in r.pdots],*[MoveToTarget(d) for d in [c_0,c_1]], run_time=0.2, rate_func=linear)
        

        self.wait(1)


        #self.play(*[GrowFromPoint(r.ldots[i],r.pdots[i]) for i in range(len(samples))], DrawBorderThenFill(loss))

        self.play(DrawBorderThenFill(loss_fml), DrawBorderThenFill(loss))
        self.wait(2)
        self.play(Uncreate(loss_fml), DrawBorderThenFill(r.ldots))
        self.play(*[MoveToTarget(d) for d in r.ldots])
        r.update(c0.get_value(),c1.get_value())
        #self.play( MoveToTarget(r.ldots))
        self.wait(1)

        #lg = VGroup(r.ldots, loss)

        self.wait(3)

        Loss.add_updater(lambda m: m.move_to(r.tdot.get_center() + UP*0.5 + LEFT*0.5))
        '''
        arrow1 = DoubleArrow(start=axes.c2p(5,-0.2), end=axes.c2p(5+(c1.get_value()-5)*0.9, -0.2))
        arrow1.add_updater(lambda m: m.put_start_and_end_on(start=axes.c2p(5,-0.2), end=axes.c2p(5+(c1.get_value()-5)*0.9, -0.2)))
        arrow2 = DoubleArrow(start=axes2.c2p(0,-0.017), end=axes2.c2p(0, -0.017)+[Loss.get_center()[0],0,0])
        arrow2.add_updater(lambda m: m.put_start_and_end_on(start=axes2.c2p(0,-0.017), end=axes2.c2p(0, -0.017)+[Loss.get_center()[0],0,0]))
        line2 = Line(start=axes2.c2p(0, -0.017)+[Loss.get_center()[0],0,0],end=Loss.get_center())
        line2.add_updater(lambda m: m.put_start_and_end_on(start=axes2.c2p(0, -0.017)+[Loss.get_center()[0],0,0],end=Loss.get_center()))
        '''


        self.play(self.camera.frame.animate.scale(1.6), run_time=1)
        self.play(self.camera.frame.animate.shift(RIGHT*5.5),FadeIn(axes2))
        self.wait(1)
        #self.play(Write(arrow1))
        #self.play(TransformFromCopy(arrow1, arrow2))
        #self.play(Write(line2))
        self.play(TransformFromCopy(r.ldots, r.tdot),MoveToTarget(r.tdot))
        self.play(Write(Loss))

        trail = TracedPath(r.tdot.get_center,time_per_anchor=0.2)
        self.add(trail)

        self.wait(3)

        while t>2:
            t-=0.5
            c1.set_value(t)
            r.update(c0.get_value(),c1.get_value())
            self.play(*[MoveToTarget(d) for d in r.qdots], *[MoveToTarget(d) for d in r.pdots],*[MoveToTarget(d) for d in r.ldots],*[MoveToTarget(d) for d in [c_0,c_1]],MoveToTarget(r.tdot),run_time=1, rate_func=linear)


        
        self.wait(3)

