using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Drawing;
using System.IO;
using Microsoft.Kinect;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.Util; 


namespace PalmDetection
{
    /// <summary>
    /// Lógica de interacción para MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        //::::::::::::::Variables:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        private KinectSensor Kinect;
        private WriteableBitmap ImagenWriteablebitmap;
        private Int32Rect WriteablebitmapRect;
        private DepthImagePixel[] DepthPixels;
        private DepthImageStream DepthStream;
        private int WriteablebitmapStride;
        private byte[] DepthImagenPixeles;
        private Image<Gray, Byte> depthFrameKinect;

        private CascadeClassifier haar; 

        private int minDepth;
        private int maxDepth;
        private string path;
        private string nombre = "1Hand";
        private bool grabar = false;
        private int i;
        //:::::::::::::fin variables::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


        //:::::::::::::Constructor::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        public MainWindow()
        {
            InitializeComponent(); 
        } 
        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


        //::::::::::::Call the methods::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            haar = new CascadeClassifier(@"C:\Users\AmericaIvone\Documents\NewHandClassifiers\OpenPalm100pos\classifier\cascade.xml"); 
            EncuentraInicializaKinect();
            CompositionTarget.Rendering += new EventHandler(CompositionTarget_Rendering);
        }
        //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        //:::::::::::::Enseguida estan los metodos para desplegar los datos de profundidad de Kinect:::::::::::::::::::::::::::::::
        private void EncuentraInicializaKinect()
        {
            Kinect = KinectSensor.KinectSensors.FirstOrDefault();

            try
            {
                if (Kinect.Status == KinectStatus.Connected)
                {
                    //Kinect.ColorStream.Enable();
                    Kinect.DepthStream.Enable();
                    Kinect.DepthStream.Range = DepthRange.Near;
                    Kinect.Start();
                }
            }
            catch
            {
                MessageBox.Show("El dispositivo Kinect no se encuentra conectado", "Error Kinect");
            }
        } //fin EncuentraKinect()    


        private void CompositionTarget_Rendering(object sender, EventArgs e)
        {
            List<Object> RoisFrames = new List<object>(2);
            Image<Bgra, Byte> imageKinectBGR;
            Image<Gray, Byte> imagenKinectGray;
            System.Drawing.Rectangle[] Rois; 

            imagenKinectGray = PollDepth();
            //imagenKinectGray = Detection(haar, imagenKinectGray); 
            RoisFrames = Detection(haar,imagenKinectGray);
            Rois = (System.Drawing.Rectangle[])RoisFrames[0];
            imagenKinectGray = (Image<Gray, Byte>)RoisFrames[1]; 
            imageKinectBGR = imagenKinectGray.Convert<Bgra, Byte>();

            DepthImage.Source = imagetoWriteablebitmap(imagenKinectGray);

            if (grabar && (i>50))
            {
                guardaimagen(imagenKinectGray, path, nombre, i-50);
                Record.IsEnabled = false;
            }
            i++;

        } //fin CompositionTarget_Rendering()  

        private Image<Gray, Byte> PollDepth()
        {
            Image<Bgra, Byte> depthFrameKinectBGR = new Image<Bgra, Byte>(640, 480);


            if (this.Kinect != null)
            {
                this.DepthStream = this.Kinect.DepthStream;
                //this.DepthValoresStream = new short[DepthStream.FramePixelDataLength];
                this.DepthPixels = new DepthImagePixel[DepthStream.FramePixelDataLength];
                this.DepthImagenPixeles = new byte[DepthStream.FramePixelDataLength * 4];
                this.depthFrameKinect = new Image<Gray, Byte>(DepthStream.FrameWidth, DepthStream.FrameHeight);

                Array.Clear(DepthImagenPixeles, 0, DepthImagenPixeles.Length);

                try
                {
                    using (DepthImageFrame frame = this.Kinect.DepthStream.OpenNextFrame(100))
                    {
                        if (frame != null)
                        {
                            frame.CopyDepthImagePixelDataTo(this.DepthPixels);

                            minDepth = 400;
                            maxDepth = 2000;

                            int index = 0;
                            for (int i = 0; i < DepthPixels.Length; ++i)
                            {
                                short depth = DepthPixels[i].Depth;

                                byte intensity = (byte)((depth >= minDepth) && (depth <= maxDepth) ? depth : 0);

                                DepthImagenPixeles[index++] = intensity;
                                DepthImagenPixeles[index++] = intensity;
                                DepthImagenPixeles[index++] = intensity;

                                ++index;
                            }

                            depthFrameKinectBGR.Bytes = DepthImagenPixeles; //The bytes are converted to a Imagen(Emgu). This to work with the functions of opencv. 
                        }
                    }
                }
                catch
                {
                    MessageBox.Show("No se pueden leer los datos del sensor", "Error");
                }
            }

            depthFrameKinect = depthFrameKinectBGR.Convert<Gray, Byte>();
            depthFrameKinect = removeNoise(depthFrameKinect, 3);

            return depthFrameKinect;
        }//fin PollDepth() 

        //::::::::::::Detection::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        private List<Object> Detection(CascadeClassifier haar, Image<Gray, Byte> frame)
        {
            List<Object> listReturn = new List<object>(2);

            if (frame != null)
            {
                System.Drawing.Rectangle[] hands = haar.DetectMultiScale(frame, 1.1, 2, new System.Drawing.Size(frame.Width / 8, frame.Height / 8), new System.Drawing.Size(frame.Width / 3, frame.Height / 3));

                foreach (System.Drawing.Rectangle roi in hands)
                {
                    Gray colorcillo = new Gray(double.MaxValue);
                    frame.Draw(roi, colorcillo, 5);
                }

                listReturn.Add(hands); 
            }

            listReturn.Add(frame);

            return listReturn;   
            //Regresa los dos valores si el frame es diferente de null, lo cual se supone que siempre es cierto, por que eso se toma en cuenta desde data poll
        }//finaliza detection()  


        //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        //:::::::::::::Method to convert a byte[] to a writeablebitmap::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        private WriteableBitmap imagetoWriteablebitmap(Image<Gray, Byte> frameHand)
        {
            Image<Bgra, Byte> frameBGR = new Image<Bgra, Byte>(DepthStream.FrameWidth, DepthStream.FrameHeight);
            byte[] imagenPixels = new byte[DepthStream.FrameWidth * DepthStream.FrameHeight];

            this.ImagenWriteablebitmap = new WriteableBitmap(DepthStream.FrameWidth, DepthStream.FrameHeight, 96, 96, PixelFormats.Bgr32, null);
            this.WriteablebitmapRect = new Int32Rect(0, 0, DepthStream.FrameWidth, DepthStream.FrameHeight);
            this.WriteablebitmapStride = DepthStream.FrameWidth * 4;

            frameBGR = frameHand.Convert<Bgra, Byte>();
            imagenPixels = frameBGR.Bytes;

            ImagenWriteablebitmap.WritePixels(WriteablebitmapRect, imagenPixels, WriteablebitmapStride, 0);

            return ImagenWriteablebitmap;
        }//end 


        //::::::::::::Method to remove the noise, using median filters::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        private Image<Gray, Byte> removeNoise(Image<Gray, Byte> imagenKinet, int sizeWindow)
        {
            Image<Gray, Byte> imagenSinRuido;

            imagenSinRuido = imagenKinet.SmoothMedian(sizeWindow);

            return imagenSinRuido;
        }//endremoveNoise 


        //:::::::::::::::::::::::Save images:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        private void stooRecord_Click(object sender, RoutedEventArgs e)
        {
            grabar = false;
        }

        private void Record_Click(object sender, RoutedEventArgs e)
        {
            grabar = true;
            i = 0;
            path = @"C:\DataBaseHand\";
        }

        private void guardaimagen(Image<Gray, Byte> imagen, string path, string nombre, int i)
        {
            imagen.Save(path + nombre + i.ToString() + ".png");
        }

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        private void Window_Unloaded(object sender, RoutedEventArgs e)
        {
            Kinect.DepthStream.Disable();
            Kinect.Stop();
        }
        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


    } //end class
}//end namespace
