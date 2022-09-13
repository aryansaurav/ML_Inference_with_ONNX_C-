// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using System.Drawing.Imaging;

namespace ConsoleApp1
{
    class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Using API");
            UseApi();
            Console.WriteLine("Done");
        }


        static void UseApi()
        {
            string modelPath = Directory.GetCurrentDirectory() + @"\testdata\model_181031_12.onnx";
            string imageFilePath = Directory.GetCurrentDirectory() + @"\testdata\sample_img.jpg";

            // Optional : Create session options and set the graph optimization level for the session
            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
            
            //using (var session = new InferenceSession(modelPath, options))
            using (var session = new InferenceSession(modelPath, SessionOptions.MakeSessionOptionWithCudaProvider()))
            {

                var inputMeta = session.InputMetadata;
                var container = new List<NamedOnnxValue>();

                //float[] inputData = LoadTensorFromFile(Directory.GetCurrentDirectory()+ @"\testdata\bench.in"); // this is the data for only one input tensor for this model
                Bitmap loaded_image = new Bitmap(imageFilePath);
                var loaded_image_tensor = ConvertImageToFloatTensorUnsafe(loaded_image);

                foreach (var name in inputMeta.Keys)
                {
                    //var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    //container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, loaded_image_tensor));
                }

                // Run the inference
                using (var results = session.Run(container))  // results is an IDisposableReadOnlyCollection<DisposableNamedOnnxValue> container
                {
                    // dump the results
                    foreach (var r in results)
                    {
                        Console.WriteLine("Output for {0}", r.Name);
                        Console.WriteLine(r.AsTensor<float>().GetArrayString());
                    }
                }
            }
        }

        static float[] LoadTensorFromFile(string filename)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }

        // Taken from stackoverflow to load image into Tensor : https://stackoverflow.com/questions/62470779/efficient-bitmap-to-onnxruntime-tensor-in-c-sharp
        static Tensor<float> ConvertImageToFloatTensorUnsafe(Bitmap image)
        {
            // Create the Tensor with the appropiate dimensions  for the NN
            Tensor<float> data = new DenseTensor<float>(new[] { 1, image.Width, image.Height, 3 });

            BitmapData bmd = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, image.PixelFormat);
            int PixelSize = 3;

            unsafe
            {
                for (int y = 0; y < bmd.Height; y++)
                {
                    // row is a pointer to a full row of data with each of its colors
                    byte* row = (byte*)bmd.Scan0 + (y * bmd.Stride);
                    for (int x = 0; x < bmd.Width; x++)
                    {
                        // note the order of colors is BGR
                        data[0, y, x, 0] = row[x * PixelSize + 2] / (float)255.0;
                        data[0, y, x, 1] = row[x * PixelSize + 1] / (float)255.0;
                        data[0, y, x, 2] = row[x * PixelSize + 0] / (float)255.0;
                    }
                }

                image.UnlockBits(bmd);
            }
            return data;
        }



    }
}
