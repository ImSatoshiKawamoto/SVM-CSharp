using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace LibPrimitiveSupportVectorMachine
{
    /// <summary>
    /// 実数の線型SVM
    /// </summary>
    public class RealValueSVM
    {
        /// <summary>
        /// Pegasousによる反復計算時間
        /// </summary>
        private double PegasousTime;

        private bool needToRecalculateW;

        private bool needToRecalculateB;

        Random r;

        private List<SignalData> dataList;

        private int dimension;

        private double eta = 0.001; // [TODO]0.001

        private double sigma;

        private RealKernelType kernel = RealKernelType.Linear;

        private double C = 100.0;

        private bool usePagasos;

        private int pegasosIterationCount;

        public RealValueSVM(int dimension, bool usePagasos, int pegasosIterationCount = 300, double sigma = 1.0)
        {
            this.PegasousTime = 0.0;
            this.needToRecalculateW = true;
            this.needToRecalculateB = true;
            this.usePagasos = usePagasos;
            this.pegasosIterationCount = pegasosIterationCount;
            this.dimension = dimension;
            this.r = new Random(1234);
            this.dataList = new List<SignalData>();
            //this.kernelType = KernelType.Linear; // kernelType;
            this.sigma = sigma;
        }

        public void AddData(MathVector x, double y)
        {
            if (x.Elements.Length != this.dimension)
            {
                throw new ApplicationException("次元数が不正です");
            }

            SignalData data = new SignalData()
            {
                Alpha = this.usePagasos ? 0.0 : this.r.NextDouble() * 10.0,
                AlphaPegasous = 0.0,
                X = x,
                Y = y
            };

            this.dataList.Add(data);
        }

        private void LearnByPegasoous(int iteration)
        {
            double lambda = 1.0;

            for (int t = 1; t <= iteration; t++)
            {
                this.PegasousTime += 1.0;
                this.LearnByPegasousInner(this.PegasousTime);
            }

            // Alphaのセット
            for (int k = 0; k < this.dataList.Count; k++)
            {
                this.dataList[k].Alpha = this.dataList[k].AlphaPegasous / lambda / this.PegasousTime;
            }

            this.needToRecalculateW = true;
            this.needToRecalculateB = true;
        }

        private void LearnByPegasousInner(double time)
        {
            double lambda = 1.0;

            int last = this.dataList.Count - 1;

            int t = this.r.Next(0, last);
            double coop = 1.0 / lambda / time;

            double y1 = this.dataList[t].Y;
            MathVector x1 = this.dataList[t].X;
            double total = 0.0;

            for (int k = 0; k < this.dataList.Count; k++)
            {
                if(k == t) { continue; }

                double y2 = this.dataList[k].Y;
                MathVector x2 = this.dataList[k].X;
                double alpha2 = this.dataList[k].AlphaPegasous;
                double work = 0.0;

                if (this.kernel == RealKernelType.Linear)
                {
                    work = y1 * coop * y2 * alpha2 * x1 * x2;
                }
                else
                {
                    work = y1 * coop * y2 * alpha2 * this.GaussKernel(x1, x2);
                }

                total += work;
            }

            if(total < 1)
            {
                this.dataList[t].AlphaPegasous += 1.0;
            }
        }

        /// <summary>
        /// マルチスレッドで学習処理を実行します
        /// </summary>
        /// <param name="threadCount"></param>
        public void LearnThreading(int threadCount)
        {
            if (this.usePagasos)
            {
                this.LearnByPegasoous(this.pegasosIterationCount);
            }
            else
            {
                var taskList = new Task[threadCount];

                for (var i = 0; i < threadCount; i++)
                {
                    var threadIndex = i;

                    taskList[threadIndex] = Task.Factory.StartNew(() =>
                    {
                        this.LearnPartial(threadIndex, threadCount);
                    });
                }

                Task.WaitAll(taskList);
            }

            this.needToRecalculateW = true;
            this.needToRecalculateB = true;
        }

        private void LearnPartial(int threadIndex, int threadCount)
        {
            for (int k = 0; k < this.dataList.Count; k++)
            {
                if (k % threadCount != threadIndex)
                {
                    continue;
                }

                {
                    double dLdA = 0.0;

                    for (int j = 0; j < this.dataList.Count; j++)
                    {
                        double alpha = this.dataList[j].Alpha;
                        double yk = this.dataList[k].Y;
                        double yj = this.dataList[j].Y;

                        MathVector xk = this.dataList[k].X;
                        MathVector xj = this.dataList[j].X;

                        double dLdA1 = -alpha * yk * yj * xk * xj;
                        //double dLdA1 = -alpha * yk * yj * this.KernelFunc(xk, xj);

                        switch (this.kernel)
                        {
                            case RealKernelType.Linear:
                                {
                                    dLdA1 = -alpha * yk * yj * xk * xj;
                                }
                                break;

                            case RealKernelType.Gaussian:
                                {
                                    dLdA1 = -alpha * yk * yj * GaussKernel(xk, xj);
                                }
                                break;
                        }

                        dLdA += dLdA1;
                    }

                    dLdA += 1.0;

                    this.dataList[k].Alpha += this.eta * dLdA;

                    if (this.dataList[k].Alpha < 0)
                    {
                        this.dataList[k].Alpha = 0.0; // サポートベクターではない
                    }

                    if (this.dataList[k].Alpha > this.C)
                    {
                        this.dataList[k].Alpha = this.C; // ソフトマージンのパラメータ
                    }
                }
            }
        }

        public void NormalizeAlpha()
        {
            double X = 0.0;
            double N = 0.0;

            for (int k = 0; k < this.dataList.Count; k++)
            {
                double alpha = this.dataList[k].Alpha;
                double y = this.dataList[k].Y;

                // 非サポートベクターの除外
                if (alpha == 0.0) { continue; }

                X += alpha * y;
                N += 1.0;
            }

            double XMean = X / N;

            for (int k = 0; k < this.dataList.Count; k++)
            {
                double alpha = this.dataList[k].Alpha;
                double alpha2 = alpha - this.dataList[k].Y * XMean;

                // 非サポートベクターの除外
                if (alpha == 0.0) { continue; }

                this.dataList[k].Alpha = Math.Max(alpha2, 0.0);
            }
        }

        private double b = 0.0;

        /// <summary>
        /// オフセットを取得(再計算の必要がある場合は再計算します)
        /// </summary>
        /// <returns></returns>
        private double B()
        {
            if (!this.needToRecalculateB)
            {
                return this.b;
            }

            double result = 0.0;


            switch (this.kernel)
            {
                case RealKernelType.Linear:
                    {
                        MathVector w = this.W();
                        double N = 0.0;

                        for (int k = 0; k < this.dataList.Count; k++)
                        {
                            if (this.dataList[k].Alpha > 0)
                            {
                                // サポートベクターからオフセットbを逆算
                                double b = w * this.dataList[k].X - this.dataList[k].Y;

                                result += b;
                                N += 1.0;
                            }
                        }

                        result /= N;
                        this.b = result;
                    }
                    break;

                case RealKernelType.Gaussian:
                    {
                        double N = 0.0;

                        for (int k = 0; k < this.dataList.Count; k++)
                        {
                            if (this.dataList[k].Alpha > 0)
                            {
                                // サポートベクターからオフセットbを逆算
                                double wx = 0;

                                foreach (SignalData data in this.dataList)
                                {
                                    if (data.Alpha > 0)
                                    {
                                        wx += data.Alpha * data.Y * GaussKernel(data.X, this.dataList[k].X);
                                    }
                                }

                                double b = wx - this.dataList[k].Y;
                                result += b;
                                N += 1.0;
                            }

                            result /= N;
                            this.b = result;
                        }
                    }
                    break;
            }

            this.needToRecalculateB = false;

            return result;
        }

        private MathVector w = null;

        /// <summary>
        /// 識別関数の係数を求めます
        /// </summary>
        /// <returns></returns>
        public MathVector W()
        {
            // 計算済み
            if (!this.needToRecalculateW)
            {
                return this.w;
            }

            List<double> list = new List<double>();
            for (int n = 0; n < this.dimension; n++)
            {
                list.Add(0.0);
            }

            MathVector w = new MathVector(list);

            foreach (SignalData data in this.dataList)
            {
                w += data.Alpha * data.Y * data.X;
            }

            this.w = w;
            this.needToRecalculateW = false;

            return w;
        }

        public bool Classify(MathVector x)
        {
            double f = 0.0;

            switch (this.kernel)
            {
                case RealKernelType.Linear:
                    if (this.usePagasos)
                    {
                        // f = this.W() * x;
                        f = this.W() * x - this.B();
                    }
                    else
                    {
                        f = this.W() * x - this.B();
                    }

                    break;

                case RealKernelType.Gaussian:
                    {
                        double wx = 0;

                        foreach (SignalData data in this.dataList)
                        {
                            if (data.Alpha > 0)
                            {
                                wx += data.Alpha * data.Y * GaussKernel(data.X, x);
                            }
                        }

                        if (this.usePagasos)
                        {
                            // f = wx;
                            f = wx - this.B();
                        }
                        else
                        {
                            f = wx - this.B();
                        }
                    }

                    break;
            }

            //Console.WriteLine(string.Format("\t\tRe(f) : {0}, Im(f) : {1}", f.Real, f.Im));

            if (f > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        private double GaussKernel(MathVector x, MathVector mu)
        {
            MathVector dif = x - mu;
            double wa = (dif * dif) * (-1.0) / this.sigma / this.sigma;
            double result =Math.Exp(wa);
            return result;
        }

        [Obsolete("This method doesn't work when RBF Kernel is used.")]
        public double LossFunc()
        {
            return (this.W() * this.W());
        }

        /// <summary>
        /// サポートベクター数を取得します
        /// </summary>
        /// <returns></returns>
        public int SupportVectorCount()
        {
            int result = 0;

            foreach (SignalData data in this.dataList)
            {
                if(data.Alpha > 0)
                {
                    result++;
                }
            }

            return result;
        }

        public enum RealKernelType : int
        {
            /// <summary>
            /// 線型SVM
            /// </summary>
            Linear = 1,

            /// <summary>
            /// ガウスカーネル
            /// </summary>
            Gaussian = 2,
        }

        public class SignalData
        {
            /// <summary>
            /// ラグランジュ係数
            /// </summary>
            public double Alpha { get; set; }

            public double AlphaPegasous { get; set; }

            /// <summary>
            /// 入力ベクトル
            /// </summary>
            public MathVector X { get; set; }

            /// <summary>
            /// 分類結果(1 or -1)
            /// </summary>
            public double Y { get; set; }
        }
    }
}
