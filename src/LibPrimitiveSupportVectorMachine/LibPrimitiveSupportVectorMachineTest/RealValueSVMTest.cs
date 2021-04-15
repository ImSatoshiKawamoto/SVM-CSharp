using LibPrimitiveSupportVectorMachine;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace LibPrimitiveSupportVectorMachineTest
{
    [TestClass]
    public class RealValueSVMTest
    {
        [TestMethod]
        public void GradientDescentOptimizeTest()
        {
            RealValueSVM svm = new RealValueSVM(2, RealValueSVM.RealKernelType.Linear);

            List<double> list1 = new List<double>();
            list1.Add(1.0);
            list1.Add(1.5);

            List<double> list2 = new List<double>();
            list2.Add(3.8);
            list2.Add(5.2);

            List<double> list3 = new List<double>();
            list3.Add(0.8);
            list3.Add(1.2);

            MathVector vec1 = new MathVector(list1);
            svm.AddData(vec1, 1.0);

            MathVector vec2 = new MathVector(list2);
            svm.AddData(vec2, -1.0);

            MathVector vec3 = new MathVector(list3);
            svm.AddData(vec3, 1.0);

            List<MathVector> xList = new List<MathVector>();
            xList.Add(vec1);
            xList.Add(vec2);
            xList.Add(vec3);

            svm.CreatePredictCache(xList);

            for (int n = 0; n < 5000; n++)
            {
                System.Diagnostics.Trace.WriteLine(string.Format("n:{0}", n));

                svm.NormalizeAlpha();
                svm.LearnThreading(1);

                bool a = svm.Classify(vec1);
                bool b = svm.Classify(vec2);
                bool c = svm.Classify(vec3);

                System.Diagnostics.Trace.WriteLine(string.Format("\t(—\‘ª:³‰ð), {0}:true, {1}:false, {2}:true, Loss:{3}", a, b, c, svm.LossFunc()));
            }
        }

        [TestMethod]
        public void PegasosOptimizeTest()
        {
            RealValueSVM svm = new RealValueSVM(2, RealValueSVM.RealKernelType.Gaussian);

            List<double> list1 = new List<double>();
            list1.Add(1.0);
            list1.Add(1.5);

            List<double> list2 = new List<double>();
            list2.Add(3.8);
            list2.Add(5.2);

            List<double> list3 = new List<double>();
            list3.Add(0.8);
            list3.Add(1.2);

            MathVector vec1 = new MathVector(list1);
            svm.AddData(vec1, 1.0);

            MathVector vec2 = new MathVector(list2);
            svm.AddData(vec2, -1.0);

            MathVector vec3 = new MathVector(list3);
            svm.AddData(vec3, 1.0);

            List<MathVector> xList = new List<MathVector>();
            xList.Add(vec1);
            xList.Add(vec2);
            xList.Add(vec3);

            svm.CreatePredictCache(xList);

            for (int n = 0; n < 2000; n++)
            {
                System.Diagnostics.Trace.WriteLine(string.Format("n:{0}", n));

                // svm.NormalizeAlpha(); // [TODO]ŠwKŒã‚ÉŽÀŽ{??
                // svm.LearnByPegasoous(1); // 1000‚Æ‚©‚É
                svm.LearnThreading(10);

                bool a = svm.Classify(vec1);
                bool b = svm.Classify(vec2);
                bool c = svm.Classify(vec3);

                System.Diagnostics.Trace.WriteLine(string.Format("\t(—\‘ª:³‰ð), {0}:true, {1}:false, {2}:true, Loss:{3}", a, b, c, svm.LossFunc()));
            }
        }
    }
}
