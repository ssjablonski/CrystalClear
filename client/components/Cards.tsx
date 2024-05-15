// import Image from 'next/image';
// import React from 'react';

// interface CardProps {
//   imgSrc: string;
//   name: string;
//   description: string;
// }

// export const Card: React.FC<CardProps> = ({ src, name, description }) => {
//   return (
//     <div className="max-w-sm rounded overflow-hidden shadow-lg p-4 m-4 bg-lightSecondary">
//       <Image className="w-full rounded-xl" src={src} alt={`The gemstone ${name}`} width={200} height={200} />
//       <div className="px-6 py-4 flex flex-col">
//         <div className="font-bold text-xl mb-2 mx-auto">{name}</div>
//         <p className="text-gray-700 text-base">
//           {description}
//         </p>
//       </div>
//     </div>
//   );
// };
    


// import { useState } from "react";
// import { motion, AnimatePresence } from "framer-motion";


// export default function Cards({ tab }) {
//   const [selectedTab, setSelectedTab] = useState(tab[0]);
//   // console.log("tab w cards",tab)
//   return (
//     <div className="rounded-xl bg-white overflow-hidden flex flex-col ">
//       <nav className="bg-secondary rounded-lg rounded-b-none p-2">
//         <ul className="list-none flex w-full">
//           {Array.isArray(tab) && tab.map((item) => (
//             <li
//               key={item.name}
//               className={item === selectedTab ? "list-none rounded-lg rounded-b-none p-2 bg-lightSecondary flex justify-between items-center flex-1 cursor-pointer" : "list-none rounded-lg rounded-b-none p-2 bg-white flex justify-between items-center flex-1 cursor-pointer"}
//               onClick={() => setSelectedTab(item)}
//             >
//               {`${item.name}`}
//               {item === selectedTab ? (
//                 <motion.div className="underline bg-blue-400" layoutId="underline" />
//               ) : null}
//             </li>
//           ))}
//         </ul>
//       </nav>
//       <main className="flex justify-center items-center flex-grow">
//         <AnimatePresence mode="wait">
//           <motion.div
//             key={selectedTab ? selectedTab.label : "empty"}
//             initial={{ y: 10, opacity: 0 }}
//             animate={{ y: 0, opacity: 1 }}
//             exit={{ y: -10, opacity: 0 }}
//             transition={{ duration: 0.2 }}
//           >
//             {selectedTab ? selectedTab : "😋"}
//           </motion.div>
//         </AnimatePresence>
//       </main>
//     </div>
//   );
// }


import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Image from 'next/image';

export default function Cards({ tab }) {
  const [selectedTab, setSelectedTab] = useState(tab[0]);
  
  return (
    <div className="rounded-xl bg-lightSecondary overflow-hidden flex flex-col max-w-4xl mx-auto">
      <nav className="bg-secondary rounded-lg rounded-b-none">
        <ul className="list-none flex w-full">
          {Array.isArray(tab) && tab.map((item, index) => (
            <li
              key={item.name}
              className={item === selectedTab ? "list-none rounded-lg rounded-b-none p-2 m-2 bg-darkAccent text-white flex justify-center items-center flex-1 cursor-pointer" : "list-none rounded-lg rounded-b-none p-2 m-2 bg-accent text-white flex justify-center items-center flex-1 cursor-pointer"}
              onClick={() => setSelectedTab(item)}
            >
              {`${item.name}`}
            </li>
          ))}
        </ul>
      </nav>
      <main className="flex justify-center items-center flex-grow">
          {selectedTab && (
            <motion.div
              key={selectedTab.name}
              initial={{  opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{  opacity: 0 }}
              transition={{ duration: 0.7 }}
              className="flex items-center p-6"
            >
              <Image src={selectedTab.img} alt={selectedTab.name} height={300} width={300} className="p-2 rounded-2xl mb-4" />
              <div className='flex flex-col items-center'>
                <h2 className="text-xl font-semibold mb-2">{selectedTab.name}</h2>
                <p className="text-gray-600 text-center w-2/3">{selectedTab.description}</p>
              </div>
                
            </motion.div>
          )}
      </main>
    </div>
  );
}
