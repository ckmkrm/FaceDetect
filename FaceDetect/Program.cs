using System;
using OpenCvSharp;

class Program
{
    static void Main()
    {
        // 0 = ilk webcam. Laptopta tek kamera varsa 0 yeter.
        using var capture = new VideoCapture(0);
        if (!capture.IsOpened())
        {
            Console.WriteLine("Kamera açılamadı. Cihaz/permission/driver kontrol et.");
            return;
        }

        // Yüz sınıflandırıcı (XML dosyanı proje çalıştırma klasörüne koy).
        using var faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
        if (faceCascade.Empty())
        {
            Console.WriteLine("Cascade yüklenemedi. XML dosyasını doğru yere koyduğundan emin ol.");
            return;
        }

        using var window = new Window("Face detection - Esc ile çık");
        using var frame = new Mat();

        Console.WriteLine("Algılama başladı. Esc ile çık.");

        while (true)
        {
            capture.Read(frame);
            if (frame.Empty()) break;

            // Griye çevir ve histogram eşitleyebilirsin (bazı ışık koşulları için faydalı)
            using var gray = new Mat();
            Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);
            Cv2.EqualizeHist(gray, gray);

            // Yüzleri bul
            var faces = faceCascade.DetectMultiScale(
                image: gray,
                scaleFactor: 1.1,
                minNeighbors: 4,
                flags: HaarDetectionTypes.ScaleImage,
                minSize: new Size(30, 30)
            );

            // Kutu çiz
            foreach (var r in faces)
            {
                Cv2.Rectangle(frame, r, Scalar.Red, 2);
            }

            window.ShowImage(frame);

            var key = Cv2.WaitKey(1);
            if (key == 27) // ESC
                break;
        }

        capture.Release();
        Cv2.DestroyAllWindows();
    }
}