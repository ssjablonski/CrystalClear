"use client";

import React, { useEffect, useState } from "react";
import axios from "axios";
import Image from "next/image";
import Navbar from "@/components/Navbar";
import { useRouter } from "next/router";
import Link from "next/link";
import { redirect } from "next/navigation";
import { useData } from "../contexts/ResultsContext";

const Prediction: React.FC = () => {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState<string[] | null>(null);
  const fileInputRef = React.createRef();
  const {data, addData, removeData, setSrc} = useData();

  useEffect(() => {
    if (prediction) {
      redirect("/results");
    }
  }, [prediction]);


  const handleFileChange = (e) => {
    setFile(e.target.files[0]);

    let reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(e.target.files[0]);
  };

  const handleReset = () => {
    setFile(null);
    setImagePreview(null);
    setPrediction(null);
    removeData();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setPrediction(response.data.prediction);
      addData(response.data.prediction);
      setSrc(imagePreview)
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };


  return (
    <div>
        <Navbar />
        <div className="flex flex-col">
            <h1 className="text-4xl mx-auto py-4">Recognize your stone now!</h1>
            <p className="text-2xl mx-auto pt-2 pb-6">With just a simple click you can discover informations about your stone!</p>
            {imagePreview && (
                <div>
                    <h1 className="text-xl flex justify-center">File that you uploaded:</h1>
                    <Image
                        className="object-cover rounded-xl mt-4 mx-auto max-w-md max-h-96"
                        src={imagePreview}
                        alt="Preview"
                        width={400}
                        height={400}
                    />
                </div>
            )}
            <form onSubmit={handleSubmit} className="mx-auto">
                <div className="flex flex-col py-4 rounded">
                    <label className="flex-1 w-full flex flex-col items-center mb-2 px-4 py-4 bg-accent text-white rounded-lg tracking-wide uppercase cursor-pointer hover:bg-darkAccent">
                        <span className="mt-2 text-base leading-normal">Select a file</span>
                        <input type="file" className="hidden" onChange={handleFileChange}/>
                    </label>
                    <div className="flex">
                        <button className="flex-1 mr-2 bg-accent text-white rounded-xl p-4 uppercase cursor-pointer flex-grow" type="submit">
                            Upload and Predict
                        </button>
                        <button className="flex-1 bg-accent text-white rounded-xl p-4 uppercase cursor-pointer" onClick={() => handleReset()}>
                            Reset
                        </button>
                    </div>
                </div>
            </form>
        </div>
    </div>
  );
}

export default Prediction;