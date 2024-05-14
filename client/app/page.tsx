import Navbar from "@/components/Navbar";
import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div>
        <Navbar />
        <div className="text-center py-12 max-w-5xl mx-auto">
            <h1 className="text-4xl font-bold mb-8">Discover the World of Gemstones with Our App!</h1>
            <p className="text-xl mb-6 mx-24">Our advanced gemstone recognition app allows you to quickly and accurately identify precious minerals. Whether you're a collector, jeweler, or geology enthusiast, our technology will help you uncover the beauty hidden in every stone.</p>
            <ul className="list-disc list-inside text-left max-w-md mx-auto mb-10">
                <li className='pb-2 text-lg'><strong>Fast and Accurate Identification</strong>: Utilize our advanced image recognition model to instantly identify any gemstone.</li>
                <li className='pb-2 text-lg'><strong>Ease of Use</strong>: An intuitive user interface makes gemstone identification simple and enjoyable.</li>
                <li className='pb-2 text-lg'><strong>Extensive Database</strong>: Access detailed information on hundreds of different gemstones.</li>
                <li className='pb-2 text-lg'><strong>Mobility</strong>: Use the app anywhere, anytime, with support for mobile devices.</li>
            </ul>
            <Link href="/predictions" className="bg-accent text-white py-4 px-6 rounded-lg hover:bg-darkAccent transition">
                Start Your Gemstone Journey Today!
            </Link>
        </div>
    </div>

  )
}
