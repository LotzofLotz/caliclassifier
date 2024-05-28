import { StatusBar } from "expo-status-bar";
import { StyleSheet, Text, View, Button } from "react-native";
import { Profiler, useEffect, useState } from "react";
import { Accelerometer, Gyroscope } from "expo-sensors";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import * as FileSystem from "expo-file-system";
import * as Asset from "expo-asset";

export default function App() {
  const [isTracking, setIsTracking] = useState(false);
  const [accData, setAccData] = useState([]);
  const [gyroData, setGyroData] = useState([]);
  const [processedData, setProcessedData] = useState([]);
  const [extractedData, setExtractedData] = useState({});
  const [model, setModel] = useState(null);

  //console.log(processedData);
  console.log(extractedData);

  useEffect(() => {
    const loadModel = async () => {
      await tf.ready();
      const asset = Asset.fromModule(require("./assets/model.tflite"));
      await asset.downloadAsync();

      const modelUri = `${FileSystem.documentDirectory}model.tflite`;
      await FileSystem.copyAsync({
        from: asset.localUri,
        to: modelUri,
      });

      const tfliteModel = await tf.loadGraphModel(`file://${modelUri}`);
      setModel(tfliteModel);
      console.log("Modell geladen");
    };

    loadModel();

    let accSubscription = null;
    let gyroSubscription = null;

    if (isTracking) {
      accSubscription = Accelerometer.addListener((accelerometerData) => {
        setAccData((currentData) => [...currentData, accelerometerData]);
      });

      gyroSubscription = Gyroscope.addListener((gyroscopeData) => {
        setGyroData((currentData) => [...currentData, gyroscopeData]);
      });

      // Accelerometer.setUpdateInterval(100); // 100 ms update interval
      // Gyroscope.setUpdateInterval(100); // 100 ms update interval
    } else {
      if (accSubscription) accSubscription.remove();
      if (gyroSubscription) gyroSubscription.remove();
      if (accData.length > 11) processAccelerometerData(accData);
    }

    return () => {
      if (accSubscription) accSubscription.remove();
      if (gyroSubscription) gyroSubscription.remove();
    };
  }, [isTracking]);

  const classifyRep = async (repData) => {
    if (!model) {
      console.log("Modell ist noch nicht geladen");
      return;
    }

    const input = tf.tensor([
      [
        ...repData.accX,
        ...repData.accY,
        ...repData.accZ,
        ...repData.gyroX,
        ...repData.gyroY,
        ...repData.gyroZ,
      ],
    ]);

    const prediction = model.predict(input);
    prediction.print();
  };

  useEffect(() => {
    if (Object.keys(extractedData).length > 0) {
      classifyRep(extractedData.Rep1);
    }
  }, [extractedData]);

  const handleStartStop = () => {
    setIsTracking(!isTracking);
  };

  const minMaxScaler = (data) => {
    const min = Math.min(...data);
    const max = Math.max(...data);
    return data.map((value) => (value - min) / (max - min));
  };

  function findPeaks(
    data,
    height = [-0.4, 0],
    prominence = 0.2,
    distance = 10,
    width = 2
  ) {
    const invertedData = data.map((value) => -value);
    const peaks = [];
    const length = invertedData.length;

    for (let i = 1; i < length - 1; i++) {
      if (
        invertedData[i] > invertedData[i - 1] &&
        invertedData[i] > invertedData[i + 1]
      ) {
        const peakValue = invertedData[i];
        if (peakValue >= height[0] && peakValue <= height[1]) {
          peaks.push(i);
        }
      }
    }

    // Apply distance constraint
    const filteredPeaks = peaks.filter((peak, index, arr) => {
      if (index === 0) return true;
      return peak - arr[index - 1] >= distance;
    });

    return filteredPeaks;
  }

  const { sgg } = require("ml-savitzky-golay-generalized");

  const processAccelerometerData = (data) => {
    // Kombinieren der Accelerometer-Daten
    const combinedData = data?.map((item) => {
      if (
        item &&
        item.x !== undefined &&
        item.y !== undefined &&
        item.z !== undefined
      ) {
        const { x, y, z } = item;
        return Math.sqrt(x ** 2 + y ** 2 + z ** 2);
      } else {
        console.error("Invalid data item:", item);
        return 0; // oder irgendein anderer Standardwert
      }
    });

    // Glätten der Daten mit dem Savitzky-Golay-Filter
    const smoothedData = sgg(combinedData, {
      windowSize: 11,
      polynomial: 2,
    });

    // Normalisieren der Daten
    const normalizedData = minMaxScaler(smoothedData);

    setProcessedData(normalizedData);

    const peaks = findPeaks(normalizedData);
    // Extrahieren der Daten zwischen den Peaks
    const extractedData = {};
    for (let i = 0; i < peaks.length - 1; i++) {
      const start = peaks[i];
      const end = peaks[i + 1];

      const segmentAccX = accData.slice(start, end).map((item) => item.x);
      const segmentAccY = accData.slice(start, end).map((item) => item.y);
      const segmentAccZ = accData.slice(start, end).map((item) => item.z);
      const segmentGyroX = gyroData.slice(start, end).map((item) => item.x);
      const segmentGyroY = gyroData.slice(start, end).map((item) => item.y);
      const segmentGyroZ = gyroData.slice(start, end).map((item) => item.z);

      const processSegment = (segment) => {
        const smoothedSegment = sgg(segment, {
          windowSize: 11,
          polynomial: 2,
        });
        return minMaxScaler(smoothedSegment);
      };

      extractedData[`Rep${i + 1}`] = {
        accX: processSegment(segmentAccX),
        accY: processSegment(segmentAccY),
        accZ: processSegment(segmentAccZ),
        gyroX: processSegment(segmentGyroX),
        gyroY: processSegment(segmentGyroY),
        gyroZ: processSegment(segmentGyroZ),
      };
    }

    setExtractedData(extractedData);
  };

  return (
    <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
      <Button title={isTracking ? "Stop" : "Start"} onPress={handleStartStop} />
    </View>
  );
}
