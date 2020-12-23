using System;
using System.Collections.Generic;
using System.Text;

namespace LibPrimitiveSupportVectorMachine
{
	/// <summary>
	/// ベクトル計算をサポートするクラス
	/// </summary>
	public class MathVector
	{
		private object lockObject = new object();

		/// <summary>
		/// ベクトルの要素データ
		/// </summary>
		public double[] Elements
		{
			get;
			private set;
		}

		/// <summary>
		/// 指定次元数の0ベクトルを生成します
		/// </summary>
		/// <param name="dimension">次元数</param>
		public MathVector(int dimension)
		{
			this.Elements = new double[dimension];
			for (int n = 1; n <= dimension; n++)
			{
				this.SetElementValue(n, 0.0f);
			}
		}

		/// <summary>
		/// 初期値を指定してベクトル生成
		/// </summary>
		/// <param name="initialValueList">初期値のリスト</param>
		public MathVector(List<double> initialValueList)
		{
			this.Elements = new double[initialValueList.Count];

			for (int n = 0; n < initialValueList.Count; n++)
			{
				this.SetElementValue(n + 1, initialValueList[n]);
			}
		}

		/// <summary>
		/// ベクトルの次元
		/// </summary>
		public int Rank
		{
			get { return this.Elements.Length; }
		}

		/// <summary>
		/// ベクトルのノルム
		/// </summary>
		public double Norm
		{
			get { return this.GetNorm(); }
		}

		/// <summary>
		/// ベクトルの内積を計算します
		/// </summary>
		/// <param name="vecA">掛けられるベクトル</param>
		/// <param name="vecB">掛けるベクトル</param>
		/// <returns>内積</returns>
		public static double operator *(MathVector vecA, MathVector vecB)
		{
			double result = 0.0;

			if (vecA.Rank != vecB.Rank)
			{
				// 入力ベクトルの次元が異なる
				throw new ArgumentException();
			}

			// 内積計算
			for (int n = 1; n <= vecA.Rank; n++)
			{
				result += vecA.GetElementValue(n) * vecB.GetElementValue(n);
			}

			return result;
		}

		/// <summary>
		/// ベクトルとスカラーの積を計算します
		/// </summary>
		/// <param name="vec">掛けられるベクトル</param>
		/// <param name="x">掛けるスカラー</param>
		/// <returns>ベクトル</returns>
		public static MathVector operator *(MathVector vec, double x)
		{
			MathVector result = vec.CreateClone();

			for (int n = 1; n <= vec.Rank; n++)
			{
				result.SetElementValue(n, result.GetElementValue(n) * x);
			}

			return result;
		}

		public static MathVector operator *(double x, MathVector vec)
		{
			MathVector result = vec.CreateClone();

			for (int n = 1; n <= vec.Rank; n++)
			{
				result.SetElementValue(n, result.GetElementValue(n) * x);
			}

			return result;
		}

		/// <summary>
		/// ベクトルとスカラーの商を計算します
		/// </summary>
		/// <param name="vec">割られるベクトル</param>
		/// <param name="x">割るスカラー</param>
		/// <returns>ベクトル</returns>
		public static MathVector operator /(MathVector vec, double x)
		{
			MathVector result = vec.CreateClone();

			for (int n = 1; n <= vec.Rank; n++)
			{
				result.SetElementValue(n, result.GetElementValue(n) / x);
			}

			return result;
		}

		/// <summary>
		/// ベクトルの和を計算します
		/// </summary>
		/// <param name="vecA">足されるベクトル</param>
		/// <param name="vecB">足すベクトル</param>
		/// <returns>加算結果</returns>
		public static MathVector operator +(MathVector vecA, MathVector vecB)
		{
			MathVector result = new MathVector(vecA.Rank);

			if (vecA.Rank != vecB.Rank)
			{
				// 入力ベクトルの次元が異なる
				throw new ArgumentException();
			}

			// ベクトルの各成分の和を計算します
			for (int n = 1; n <= vecA.Rank; n++)
			{
				result.SetElementValue(n, vecA.GetElementValue(n) + vecB.GetElementValue(n));
			}

			return result;
		}

		/// <summary>
		/// ベクトルの差を計算します
		/// </summary>
		/// <param name="vecA">引かれるベクトル</param>
		/// <param name="vecB">引くベクトル</param>
		/// <returns>差を計算した結果</returns>
		public static MathVector operator -(MathVector vecA, MathVector vecB)
		{
			MathVector result = new MathVector(vecA.Rank);

			if (vecA.Rank != vecB.Rank)
			{
				// 入力ベクトルの次元が異なる
				throw new ArgumentException();
			}

			// ベクトルの各成分の差を計算します
			for (int n = 1; n <= vecA.Rank; n++)
			{
				result.SetElementValue(n, vecA.GetElementValue(n) - vecB.GetElementValue(n));
			}

			return result;
		}

		/// <summary>
		/// ベクトルの成分を取得します
		/// </summary>
		/// <param name="index">成分インデックス(1から開始することに注意)</param>
		/// <returns>ベクトルの成分</returns>
		public double GetElementValue(int index)
		{
			return this.Elements[index - 1];
		}

		/// <summary>
		/// ベクトルの成分をセットします
		/// </summary>
		/// <param name="index">ベクトル成分のインデックス(1から開始することに注意)</param>
		/// <param name="value">ベクトル成分</param>
		public void SetElementValue(int index, double value)
		{
			lock (lockObject)
			{
				this.Elements[index - 1] = value;
			}
		}

		/// <summary>
		/// ベクトルの成分を追加します
		/// </summary>
		/// <param name="index">ベクトル成分のインデックス(1から開始することに注意)</param>
		/// <param name="value">ベクトル成分</param>
		public void AddElementValue(int index, double value)
		{
			double current = this.GetElementValue(index);
			this.SetElementValue(index, value + current);
		}

		/// <summary>
		/// xとyのユークリッド距離を計算します
		/// </summary>
		/// <param name="x">入力ベクトル</param>
		/// <param name="y">入力ベクトル</param>
		/// <returns>ユークリッド距離</returns>
		public static double Distance(MathVector x, MathVector y)
		{
			// ベクトルの差を計算してノルムを返す
			MathVector dif = x - y;
			return dif.GetNorm();
		}

		/// <summary>
		/// xとyのコサイン類似度を計算します
		/// </summary>
		/// <param name="x">入力ベクトル</param>
		/// <param name="y">入力ベクトル</param>
		/// <returns>コサイン距離</returns>
		public static double Cosine(MathVector x, MathVector y)
		{
			double numer = x * y; // 内積
			double denom = x.GetNorm() * y.GetNorm();

			if (denom == 0.0)
			{
				return 1.0;
			}

			return (numer / denom);
		}

		/// <summary>
		/// ベクトルの各要素を二乗したベクトルを生成します
		/// </summary>
		/// <param name="vec">入力</param>
		/// <returns>二乗した結果</returns>
		public static MathVector CreateSquare(MathVector vec)
		{
			MathVector result = vec.CreateClone();

			for (int n = 1; n <= result.Rank; n++)
			{
				result.SetElementValue(n, result.GetElementValue(n) * result.GetElementValue(n));
			}

			return result;
		}

		/// <summary>
		/// ベクトルの各要素を指定の累乗したベクトルを生成します
		/// </summary>
		/// <param name="vec">入力</param>
		/// <param name="pow">指数部</param>
		/// <returns>累乗した結果</returns>
		public static MathVector CreatePow(MathVector vec, double pow)
		{
			MathVector result = vec.CreateClone();

			for (int n = 1; n <= result.Rank; n++)
			{
				result.SetElementValue(n, Math.Pow(result.GetElementValue(n), pow));
			}

			return result;
		}

		/// <summary>
		/// ベクトルの各要素で割ったベクトルを生成します
		/// </summary>
		/// <param name="x">割られるベクトル</param>
		/// <param name="y">割るベクトル</param>
		/// <returns>割った結果</returns>
		public static MathVector CreateDiv(MathVector x, MathVector y)
		{
			MathVector result = x.CreateClone();

			for (int n = 1; n <= x.Rank; n++)
			{
				if (y.GetElementValue(n) != 0.0)
				{
					result.SetElementValue(n, result.GetElementValue(n) / y.GetElementValue(n));
				}
			}

			return result;
		}

		/// <summary>
		/// ベクトルの各要素で割った計算結果を取得します
		/// </summary>
		/// <param name="div">割るベクトル</param>
		public void Resize(MathVector div)
		{
			for (int n = 1; n <= this.Rank; n++)
			{
				if (div.GetElementValue(n) != 0.0)
				{
					this.SetElementValue(n, this.GetElementValue(n) / div.GetElementValue(n));
				}
			}
		}

		public static MathVector Concat(MathVector x, MathVector y)
		{
			MathVector result = new MathVector(x.Rank + y.Rank);

			for (int n = 1; n <= x.Rank; n++)
			{
				result.SetElementValue(n, x.GetElementValue(n));
			}

			for (int m = x.Rank + 1; m <= y.Rank; m++)
			{
				result.SetElementValue(m + 1, y.GetElementValue(m));
			}

			return result;
		}

		/// <summary>
		/// クローンを作成
		/// </summary>
		/// <returns>成分の等しいベクトルデータ</returns>
		public MathVector CreateClone()
		{
			// rankの等しい0ベクトルを生成します
			MathVector result = new MathVector(this.Rank);

			result += this;

			return result;
		}

		/// <summary>
		/// ベクトルを文字のリストに変換
		/// </summary>
		/// <returns>文字のリスト</returns>
		public string ToJsonString()
		{
			string result = "{";

			for (int n = 1; n <= this.Rank; n++)
			{
				result += this.GetElementValue(n);
				if (n != this.Rank)
				{
					result += ",";
				}
			}

			result += "}";
			return result;
		}

		/// <summary>
		/// ベクトルのユークリッドノルムを計算します
		/// </summary>
		/// <returns>ノルム</returns>
		private double GetNorm()
		{
			double result = 0.0;

			for (int n = 1; n <= this.Rank; n++)
			{
				result += Math.Pow(this.GetElementValue(n), 2.0);
			}

			result = Math.Sqrt(result);

			return result;
		}

		/// <summary>
		/// ベクトルをdoubleのリストとして取得します
		/// </summary>
		/// <returns>ベクトル要素のリスト</returns>
		public List<double> GetElementList()
		{
			List<double> result = new List<double>();

			foreach (double element in Elements)
			{
				result.Add(element);
			}

			return result;
		}

		public static MathVector MaxPoolingAmplitude(MathVector x, MathVector y)
		{
			MathVector result = new MathVector(x.GetElementList());

			for (int n = 1; n <= x.Rank; n++)
			{
				double xElem = x.GetElementValue(n);
				double yElem = y.GetElementValue(n);

				if (Math.Abs(xElem) > Math.Abs(yElem))
				{
					result.SetElementValue(n, xElem);
				}
				else
				{
					result.SetElementValue(n, yElem);
				}
			}

			return result;
		}

		public string ElementAmplitudeToCSV()
		{
			string result = "[R]\t";

			foreach (double x in this.Elements)
			{
				result += string.Format("{0}\t", x);
			}

			return result;
		}


		public string ElementPhaseToCSV()
		{
			string result = "[Theta]\t";

			foreach (double x in this.Elements)
			{
				result += "0\t";
			}

			return result;
		}
	}
}
