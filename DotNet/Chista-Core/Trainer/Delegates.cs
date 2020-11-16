using System;
using System.Collections.Generic;
using System.Text;

namespace Photon.NeuralNetwork.Chista.Trainer
{
    public delegate void OnInitializeHandler(Instructor instructor);
    public delegate void ReflectFinishedHandler(Instructor instructor, Record record, long duration);
    public delegate void OnErrorHandler(Instructor instructor, Exception ex);
    public delegate void OnStoppedHandler(Instructor instructor);
}
