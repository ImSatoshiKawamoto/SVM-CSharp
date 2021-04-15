using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace LibPrimitiveSupportVectorMachine
{
    /// <summary>
    /// 実数の線型SVM
    /// </summary>
    public class RealValueSVM
    {

        private bool needToRecalculateW;

        private bool needToRecalculateB;

        Random r;

        private List<SignalData> dataList;

        private int dimension;

        private double eta = 0.001; // [TODO]0.001

        /// <summary>
        /// RBFカーネルのパラメータ
        /// </summary>
        public double Sigma { get; private set; }

        private RealKernelType kernel;

        /// <summary>
        /// Cパラメータ
        /// </summary>
        public double C { get; private set; }

        private readonly GradType gradType = GradType.AdaDelta;

        /// <summary>
        /// xk_xjの計算結果
        /// </summary>
        private Dictionary<(int k, int j), double> kernalCaluculatedResultCash;

        public RealValueSVM(
            int dimension,
            RealKernelType kernel,
            double C = 256,
            double sigma = 0.005)
        {
            this.needToRecalculateW = true;
            this.needToRecalculateB = true;
            this.dimension = dimension;
            this.r = new Random(1234);
            this.dataList = new List<SignalData>();
            this.kernel = kernel;
            //this.kernelType = KernelType.Linear; // kernelType;
            this.C = C;
            this.Sigma = sigma;
            this.kernalCaluculatedResultCash = new Dictionary<(int k, int j), double>();
        }

        public void GridSearchSigma()
        {

        }

        public void AddData(MathVector x, double y)
        {
            if (x.Elements.Length != this.dimension)
            {
                throw new ApplicationException("次元数が不正です");
            }

            SignalData data = new SignalData()
            {
                Alpha = this.r.NextDouble() * 10.0,
                X = x,
                Y = y,
                AlphaAdaDelta = new RealAdaDeltaParam()
            };

            this.dataList.Add(data);
        }

        /// <summary>
        /// マルチスレッドで学習処理を実行します
        /// </summary>
        /// <param name="threadCount"></param>
        public void LearnThreading(int threadCount)
        {
            if (this.kernalCaluculatedResultCash.Count == 0)
            {
                // キャッシュ構築
                Console.WriteLine("カーネル計算キャッシュ構築");

                for (int k = 0; k < this.dataList.Count; k++)
                {
                    for (int j = 0; j < this.dataList.Count; j++)
                    {
                        MathVector xk = this.dataList[k].X;
                        MathVector xj = this.dataList[j].X;

                        double xk_xj = 0.0;

                        switch (this.kernel)
                        {
                            case RealKernelType.Linear:
                                xk_xj = xk * xj;
                                break;

                            case RealKernelType.Gaussian:
                                xk_xj = GaussKernel(xk, xj);
                                break;
                        }

                        this.kernalCaluculatedResultCash[(k, j)] = xk_xj;
                    }
                }

                Console.WriteLine("カーネル計算キャッシュ構築完了");
            }

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

                        double xk_xj = this.kernalCaluculatedResultCash[(k, j)];

                        double dLdA1 = -alpha * yk * yj * xk_xj;
                        dLdA += dLdA1;
                    }

                    dLdA += 1.0;

                    double delta = 0.0;

                    switch (this.gradType)
                    {
                        case GradType.AdaDelta:
                            delta = this.dataList[k].AlphaAdaDelta.GetDelta(dLdA); // .GetDelta(dLdA.Real);
                            break;

                        case GradType.SGD:
                            delta = this.eta * dLdA; // SGD
                            break;
                    }

                    this.dataList[k].Alpha += delta; // this.eta * dLdA;

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


        private double BQuick()
        {
            if (!this.needToRecalculateB)
            {
                return this.b;
            }

            double result = 0.0;
            SignalData signal = this.dataList.Where(entity => entity.Alpha > 0).First();

            switch (this.kernel)
            {
                case RealKernelType.Linear:
                    {
                        MathVector w = this.W();

                        // サポートベクターからオフセットbを逆算
                        double b = w * signal.X - signal.Y;
                        result = b;
                    }
                    break;

                case RealKernelType.Gaussian:
                    {
                        // サポートベクターからオフセットbを逆算
                        double wx = 0;

                        foreach (SignalData data in this.dataList)
                        {
                            if (data.Alpha > 0)
                            {
                                wx += data.Alpha * data.Y * GaussKernel(data.X, signal.X);
                            }
                        }

                        double b = wx - signal.Y;
                        result = b;
                    }

                    break;
            }

            this.needToRecalculateB = false;
            this.b = result;

            return result;
        }

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

        private Dictionary<(MathVector x1, MathVector x2), double> kernelPredictCashe = new Dictionary<(MathVector x1, MathVector x2), double>();

        public void CreatePredictCache(IEnumerable<MathVector> xList)
        {
            Console.WriteLine("予測用のキャッシュ構築");

            foreach (MathVector x in xList)
            {
                foreach (SignalData data in this.dataList)
                {
                    switch (this.kernel)
                    {
                        case RealKernelType.Linear:
                            double kx = data.X * x;
                            this.kernelPredictCashe[(x, data.X)] = kx;

                            break;

                        case RealKernelType.Gaussian:
                            double kxr = GaussKernel(data.X, x);
                            this.kernelPredictCashe[(x, data.X)] = kxr;

                            break;
                    }
                }
            }

            Console.WriteLine("予測用のキャッシュ構築完了");
        }

        /// <summary>
        /// 混同行列形式の予測結果を返します
        /// </summary>
        /// <param name="xList">入力データのリスト</param>
        /// <param name="answer">正解データ</param>
        /// <param name="threadCount">スレッド数</param>
        /// <returns>混同行列</returns>
        public (int tp, int tn, int fp, int fn) PredictTest(
            IEnumerable<MathVector> xList,
            Dictionary<MathVector, bool> answer,
            int threadCount)
        {
            // データの振り分け
            List<MathVector>[] xListAry = new List<MathVector>[threadCount];
            (int tp, int tn, int fp, int fn)[] resultAry = new (int tp, int tn, int fp, int fn)[threadCount];

            for (int n = 0; n < threadCount; n++)
            {
                xListAry[n] = new List<MathVector>();
            }

            int idx = 0;

            foreach (MathVector x in xList)
            {
                xListAry[idx].Add(x);

                idx++;
                idx %= threadCount;
            }

            var taskList = new Task[threadCount];

            for (var i = 0; i < threadCount; i++)
            {
                var threadIndex = i;

                taskList[threadIndex] = Task.Factory.StartNew(() =>
                {
                    resultAry[threadIndex] = this.PredictTestInner(xListAry[threadIndex], answer);
                });
            }

            Task.WaitAll(taskList);

            int tp = 0;
            int tn = 0;
            int fp = 0;
            int fn = 0;

            foreach ((int tp, int tn, int fp, int fn) item in resultAry)
            {
                tp += item.tp;
                tn += item.tn;
                fp += item.fp;
                fn += item.fn;
            }

            return (tp, tn, fp, fn);
        }

        /// <summary>
        /// 混同行列形式の予測結果を返します
        /// </summary>
        /// <param name="xList">入力データのリスト</param>
        /// <param name="answer">正解データ</param>
        /// <returns>混同行列</returns>
        private (int tp, int tn, int fp, int fn) PredictTestInner(
            IEnumerable<MathVector> xList,
            Dictionary<MathVector, bool> answer)
        {
            int tp = 0;
            int tn = 0;
            int fp = 0;
            int fn = 0;

            foreach (MathVector x in xList)
            {
                bool predict = this.Classify(x);

                if (predict)
                {
                    if (predict == answer[x])
                    {
                        tp++;
                    }
                    else
                    {
                        fp++;
                    }
                }
                else
                {
                    if (predict == answer[x])
                    {
                        tn++;
                    }
                    else
                    {
                        fn++;
                    }
                }
            }

            if(tp==0 || tn == 0)
            {
                foreach (MathVector x in xList)
                {
                    this.Classify(x, true);
                }
            }

            return (tp, tn, fp, fn);
        }

        public bool Classify(MathVector x, bool echo = false)
        {
            double f = 0.0;
            double wx = 0.0;

            foreach (SignalData data in this.dataList)
            {
                if (data.Alpha > 0)
                {
                    (MathVector x1, MathVector x2) key = (x, data.X);
                    double kernelInnerProduct = 0.0;

                    if (this.kernelPredictCashe.ContainsKey(key))
                    {
                        kernelInnerProduct = this.kernelPredictCashe[key];
                    }
                    else
                    {
                        switch (this.kernel)
                        {
                            case RealKernelType.Linear:
                                kernelInnerProduct = data.X * x;
                                break;

                            case RealKernelType.Gaussian:
                                kernelInnerProduct = GaussKernel(data.X, x);
                                break;
                        }
                    }

                    wx += data.Alpha * data.Y * kernelInnerProduct;
                }
            }

            f = wx - this.BQuick();

            if (echo)
            {
                Console.WriteLine(string.Format("wx : {0}, b : {1}", wx, this.BQuick()));
            }

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
            double wa = (dif * dif) * (-1.0) / this.Sigma / this.Sigma;
            double result = Math.Exp(wa);
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

        public enum GradType : int
        {
            SGD = 1,
            AdaDelta = 2,
        }

        public class SignalData
        {
            /// <summary>
            /// ラグランジュ係数
            /// </summary>
            public double Alpha { get; set; }

            public RealAdaDeltaParam AlphaAdaDelta { get; set; }

            /// <summary>
            /// 入力ベクトル
            /// </summary>
            public MathVector X { get; set; }

            /// <summary>
            /// 分類結果(1 or -1)
            /// </summary>
            public double Y { get; set; }
        }

        public class RealAdaDeltaParam
        {
            /// <summary>
            /// AdaDeltaに用いる情報
            /// </summary>
            private double v;

            /// <summary>
            /// AdaDeltaに用いる情報
            /// </summary>
            private double u;

            /// <summary>
            /// AdaDeltaのハイパーパラメータ
            /// </summary>
            private readonly double rho = 0.95;

            private readonly double eps = 0.000001;

            public double GetDelta(double grad)
            {
                double v2 = this.rho * this.v + (1 - this.rho) * grad * grad;
                this.v = v2;

                double dw = Math.Sqrt(this.u + this.eps) / Math.Sqrt(this.v + this.eps) * grad;

                double u2 = this.rho * this.u + (1 - this.rho) * dw * dw;
                this.u = u2;

                return dw;
            }
        }
    }
}
