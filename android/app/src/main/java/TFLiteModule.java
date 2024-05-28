package com.caliclassifier;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Promise;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class TFLiteModule extends ReactContextBaseJavaModule {

    public TFLiteModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }

    @Override
    public String getName() {
        return "TFLite";
    }

    private MappedByteBuffer loadModelFile(String modelPath) throws IOException {
        FileInputStream inputStream = new FileInputStream(modelPath);
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = 0;
        long declaredLength = fileChannel.size();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @ReactMethod
    public void runModel(String modelPath, String inputData, Promise promise) {
        try {
            MappedByteBuffer modelFile = loadModelFile(modelPath);
            Interpreter tflite = new Interpreter(modelFile);
            // TensorFlow Lite Inferenzcode hier einfügen
            // Beispiel: float[] input = ...;
            // float[] output = new float[1];
            // tflite.run(input, output);
            // promise.resolve(output[0]);
        } catch (IOException e) {
            promise.reject("Model loading error", e);
        }
    }
}
