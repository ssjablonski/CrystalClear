'use client'
import React, { createContext, useContext, useState, ReactNode } from 'react';

// Define the type for the context state
interface ResultsContextType {
  data: string[]; // Data is now an array of strings
  src: string;
  setSrc: (src: string) => void;
  setData: (data: string[]) => void;
  addData: (item: string[]) => void; // Function to add an item to the array
  removeData: () => void; // Function to remove an item by index
}

// Create the Context with a default undefined state
const ResultsContext = createContext<ResultsContextType | undefined>(undefined);

// Create a Provider Component with proper types for its children
interface ResultsProviderProps {
  children: ReactNode;
}

export const ResultsProvider: React.FC<ResultsProviderProps> = ({ children }) => {
  const [data, setData] = useState<string[]>([]);
  const [src, setSrc] = useState<string>('');


  // Function to add an item to the data array
  const addData = (item: string[]) => {
    setData(item);
  };

  const addSrc = (src: string) => {
    setSrc(src);
  }

  // Function to remove an item from the data array by index
  const removeData = () => {
    setData([]);
    setSrc('');
  };

  // The value that will be given to the context
  const value = {
    data,
    src,
    setSrc,
    setData,
    addData,
    removeData,
  };

  return (
    <ResultsContext.Provider value={value}>
      {children}
    </ResultsContext.Provider>
  );
};

// Custom hook to use the DataContext
export const useData = (): ResultsContextType => {
  const context = useContext(ResultsContext);
  if (context === undefined) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};
