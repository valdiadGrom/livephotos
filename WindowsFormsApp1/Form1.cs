using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Cvb;
using Emgu.CV.Features2D;
using Emgu.CV.Util;
using System.IO;
using Emgu.CV.Cuda;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {




        Mat imgKeypointsModel = new Mat();
        Mat imgKeypointsTest = new Mat();
        Mat imgMatches = new Mat();
        Mat imgWarped = new Mat();
        VectorOfVectorOfDMatch filteredMatches = new VectorOfVectorOfDMatch();
        List<MDMatch[]> filteredMatchesList = new List<MDMatch[]>();

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        
        private void OpenFileDialog1_FileOk(object sender, CancelEventArgs e)
        {
            
        }

        private void Button1_Click(object sender, EventArgs e)
        {
            Image<Rgb, Byte> image;

            ORBDetector detector = new ORBDetector();
            BFMatcher matcher = new BFMatcher(DistanceType.Hamming2);
            
            OpenFileDialog open = new OpenFileDialog();
            open.Filter = "Image Files (*.tif; *.dcm; *.jpg; *.jpeg; *.bmp)|*.tif; *.dcm; *.jpg; *.jpeg; *.bmp";

            if (open.ShowDialog() == DialogResult.OK)
            {
                image = new Image<Rgb, Byte>(open.FileName);
                (Image<Rgb, byte> Image, VectorOfKeyPoint Keypoints, Mat Descriptors) imgModel = (image.Resize(0.2, Emgu.CV.CvEnum.Inter.Area), new VectorOfKeyPoint(), new Mat());

                imageBox1.Image = image;
                //detector.DetectAndCompute(imgModel.Image, null, imgModel.Keypoints, imgModel.Descriptors, false);
                //matcher.Add(imgModel.Descriptors);
                //matcher.KnnMatch(imgTest.Descriptors, matches, 1, null);
                //imageBox2.Image = imgModel.Image;

                var ext = new List<string> { ".jpg", ".gif", ".png" };
                var myFiles = Directory.GetFiles(@"\\psta.ru\EDU\Students\s43880\Desktop\!Примеры фото и скриптов для живых рисунков\TEmplates", "*.*", SearchOption.AllDirectories)
                     .Where(s => ext.Contains(Path.GetExtension(s)));
                int Max = 0;
                string MaxPath = "000";
                var scene = new Mat(open.FileName);
                foreach (var a in myFiles)
                {
                    var model = new Mat(a);

                    VectorOfKeyPoint modelKeyPoints;
                    VectorOfKeyPoint observedKeyPoints;
                    VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
                    Mat mask;
                    Mat homography;
                    int Match = FindMatch1(model, scene, out modelKeyPoints, out observedKeyPoints, matches, out mask, out homography);
                    if (Match > Max)
                    {
                        Max = Match;
                        MaxPath = a;
                    }
                }
                //var model = new Mat(@"\\psta.ru\EDU\Students\s43880\Desktop\!Примеры фото и скриптов для живых рисунков\TEmplates\traktor.jpg");
                //var scene = new Mat(open.FileName);
                label1.Text = Path.GetFileName(MaxPath);
                var result = Draw(new Mat(MaxPath), scene, 1);
                imageBox2.Image = result;
                var result2 = Draw(new Mat(MaxPath), scene, 2);
                imageBox3.Image = result2;


            }


        }
        public static void FindMatch(Mat modelImage, Mat observedImage, out VectorOfKeyPoint modelKeyPoints, out VectorOfKeyPoint observedKeyPoints, VectorOfVectorOfDMatch matches, out Mat mask, out Mat homography)
        {
            int k = 2;
            double uniquenessThreshold = 0.80;
            homography = null;
            modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();
            using (UMat uModelImage = modelImage.GetUMat(AccessType.Read))
            using (UMat uObservedImage = observedImage.GetUMat(AccessType.Read))
            {
                var featureDetector = new ORBDetector(9000);
                Mat modelDescriptors = new Mat();
                featureDetector.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);
                Mat observedDescriptors = new Mat();
                featureDetector.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);
                using (var matcher = new BFMatcher(DistanceType.Hamming, false))
                {
                    matcher.Add(modelDescriptors);

                    matcher.KnnMatch(observedDescriptors, matches, k, null);
                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                    int nonZeroCount = CvInvoke.CountNonZero(mask);
                    if (nonZeroCount >= 4)
                    {
                        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                            matches, mask, 1.5, 20);
                        if (nonZeroCount >= 4)
                            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints,
                                observedKeyPoints, matches, mask, 2);
                    }
                }
            }
        }
        public int  FindMatch1(Mat modelImage, Mat observedImage, out VectorOfKeyPoint modelKeyPoints, out VectorOfKeyPoint observedKeyPoints, VectorOfVectorOfDMatch matches, out Mat mask, out Mat homography)
        {
            int k = 2;
            int nonZeroCount = 0;
            double uniquenessThreshold = 0.80;
            homography = null;
            modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();
            using (UMat uModelImage = modelImage.GetUMat(AccessType.Read))
            using (UMat uObservedImage = observedImage.GetUMat(AccessType.Read))

            {
                var featureDetector = new ORBDetector(9000);
                Mat modelDescriptors = new Mat();
                featureDetector.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);
                Mat observedDescriptors = new Mat();
                featureDetector.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);
                using (var matcher = new BFMatcher(DistanceType.Hamming, false))
                {
                    matcher.Add(modelDescriptors);

                    matcher.KnnMatch(observedDescriptors, matches, k, null);
                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                    nonZeroCount = CvInvoke.CountNonZero(mask);
                    if (nonZeroCount >= 4)
                    {
                        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                            matches, mask, 1.5, 20);
                        //if (nonZeroCount >= 4)
                        //    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints,
                        //        observedKeyPoints, matches, mask, 2);
                    }
                }
                
            }
            return nonZeroCount;
        }

        //private void Matching()
        //{

        //    Image<Bgr, Byte> modelImage = new Image<Bgr, byte>(@"C:\Users\rkosharil\Desktop\Camera Reasearch (EMGU CV)\Matching\Matching\model\" + "1" + ".png");
        //    Image<Bgr, Byte> observedImage = new Image<Bgr, byte>(@"C:\Users\rkosharil\Desktop\Camera Reasearch (EMGU CV)\Matching\Matching\scene\" + "1" + ".png");
        //    Image<Bgr, Byte> imgshow = observedImage.Copy();

        //    double[] minValues, maxValues;
        //    Point[] minLocations, maxLocations;

        //    using (var result = observedImage.MatchTemplate(modelImage, Emgu.CV.CvEnum.TM_TYPE.CV_TM_SQDIFF_NORMED))
        //    {
        //        result.MinMax(out minValues, out maxValues, out minLocations, out maxLocations);

        //        if (maxValues[0] > 0.95)
        //        {
        //            var match = new Rectangle(maxLocations[0], modelImage.Size);
        //            imgshow.Draw(match, new Bgr(Color.Red), 3);
        //            //textBox1.Text = match.X.ToString();
        //            //textBox2.Text = match.Y.ToString();
        //        }
        //        else
        //        {
        //            MessageBox.Show("Match Not Detected");
        //        }
        //    }

        //    //pictureBox2.Image = imgshow.Bitmap;
        //    imageBox4.Image = imgshow;
        //}
        public static Mat Draw(Mat modelImage, Mat observedImage,int no)
        {
            Mat homography;
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
            {
                Mat mask;
                FindMatch(modelImage, observedImage, out modelKeyPoints, out observedKeyPoints, matches, out mask, out homography);
                Mat result = new Mat();
                Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
                    matches, result, new MCvScalar(255, 0, 0), new MCvScalar(0, 0, 255), mask);

                if (homography != null)
                {
                    var imgWarped = new Mat();
                    CvInvoke.WarpPerspective(observedImage, imgWarped, homography, modelImage.Size, Inter.Linear, Warp.InverseMap);
                    Rectangle rect = new Rectangle(Point.Empty, modelImage.Size);
                    var pts = new PointF[]
                    {
                  new PointF(rect.Left, rect.Bottom),
                  new PointF(rect.Right, rect.Bottom),
                  new PointF(rect.Right, rect.Top),
                  new PointF(rect.Left, rect.Top)
                    };

                    pts = CvInvoke.PerspectiveTransform(pts, homography);
                    var points = new Point[pts.Length];
                    for (int i = 0; i < points.Length; i++)
                        points[i] = Point.Round(pts[i]);
                    if (no == 1){
                        using (var vp = new VectorOfPoint(points))
                        {
                            CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                        }
                        return result;

                    }
                    if (no == 2)
                    {
                        using (var vp = new VectorOfPoint(points))
                        {
                            CvInvoke.WarpPerspective(observedImage, result, homography, modelImage.Size, Inter.Linear, Warp.InverseMap);
                        }
                    }
                }
                return result;
            }
        }
    }
    
}
