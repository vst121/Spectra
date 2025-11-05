namespace Spectra.ImageProcessing.Processing;

using OpenCvSharp;
using OpenCvSharp.Dnn;

public class AgeGenderOpenCvService : IDisposable
{
    public AgeGenderOpenCvService()
    {
    }

    public void PredictAgeGender(Stream imageStream)
    {
        var path = @"D:\DotNetProjects10\Spectra\Spectra.ImageProcessing\Models\AgeGenderOpenCV\";

        string ageProto = Path.Combine(path, "age_deploy.prototxt");
        string ageModel = Path.Combine(path, "age_net.caffemodel");
        string genderProto = Path.Combine(path, "gender_deploy.prototxt");
        string genderModel = Path.Combine(path, "gender_net.caffemodel");
        var faceCascade = new CascadeClassifier(Path.Combine(path, "haarcascade_frontalface_default.xml"));

        string[] ageList = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"];
        string[] genderList = ["Male", "Female"];

        var ageNet = CvDnn.ReadNetFromCaffe(ageProto, ageModel);
        var genderNet = CvDnn.ReadNetFromCaffe(genderProto, genderModel);

        //// Load the image
        //var imagePath = @"D:\DotNetProjects10\Spectra\Spectra.BlazorApp\TestImages\";
        //Mat frame = Cv2.ImRead(imagePath + "test.jpg");

        byte[] imageBytes;
        using (var ms = new MemoryStream())
        {
            imageStream.CopyTo(ms);
            imageBytes = ms.ToArray();
        }

        using Mat imgMat = Cv2.ImDecode(imageBytes, ImreadModes.Color);
        Mat frame = imgMat.Clone();

        var faces = faceCascade.DetectMultiScale(frame, 1.1, 5, HaarDetectionTypes.ScaleImage);

        foreach (var face in faces)
        {
            Mat faceROI = new(frame, face);
            Mat blob = CvDnn.BlobFromImage(faceROI, 1.0, new Size(227, 227),
                                           new Scalar(78.4263377603, 87.7689143744, 114.895847746), false);
            //Mat blob = CvDnn.BlobFromImage(faceROI, 1.0, new Size(227, 227), new Scalar(104, 117, 123), false);

            // Predict gender
            genderNet.SetInput(blob);
            var genderPreds = genderNet.Forward();

            OpenCvSharp.Point minLoc, maxLoc;
            double minVal, maxVal;

            Cv2.MinMaxLoc(genderPreds, out minVal, out maxVal, out minLoc, out maxLoc);
            int genderIndex = (int)maxLoc.X;
            string gender = genderList[genderIndex];

            // Predict age
            ageNet.SetInput(blob);
            var agePreds = ageNet.Forward();

            Cv2.MinMaxLoc(agePreds, out minVal, out maxVal, out minLoc, out maxLoc);
            int ageIndex = (int)maxLoc.X;
            string age = ageList[ageIndex];

            // Draw result
            Cv2.PutText(frame, $"{gender}, {age}", new Point(face.X, face.Y - 10),
                        HersheyFonts.HersheySimplex, 0.8, Scalar.Red, 2);
            Cv2.Rectangle(frame, face, Scalar.Green, 2);
        }

        Cv2.ImShow("Age and Gender Detection", frame);
        Cv2.WaitKey(0);
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
    }
}