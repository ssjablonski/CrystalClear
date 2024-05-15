'use client'

import Navbar from '@/components/Navbar';
import React, {useContext} from 'react'
import { useData } from '../contexts/ResultsContext';
import Image from 'next/image';
import Cards, { Card } from '@/components/Cards';
import Link from 'next/link';

interface ResultsProps {
    props: string[]
}

const predictionMapping = {
    "ametrine": {
        "name": "Ametrine",
        "img": "/ametrine.jpg",
        "description": "Ametrine, also known as trystine or by its trade name as bolivianite, is a naturally occurring variety of quartz. It is a mixture of amethyst and citrine with zones of purple and yellow or orange. Almost all commercially available ametrine is mined in Bolivia."
    },
    "amethyst": {
        "name": "Amethyst",
        "img": "/amethyst.jpg",
        "description": "Amethyst is a violet variety of quartz. The name comes from the Koine Greek amethystos which translates to intoxicate, a reference to the belief that the stone protected its owner from drunkenness. Ancient Greeks wore amethyst and carved drinking vessels from it in the belief that it would prevent intoxication.Amethyst, a semiprecious stone, is often used in jewelry."
    },
    "aquamarine": {
        "name": "Aquamarine",
        "img": "/aquamarine.jpg",
        "description": "Aquamarine is a pale-blue to light-green variety of the beryl family, with its name relating to water and sea. The color of aquamarine can be changed by heat, with a goal to enhance its physical appearance (though this practice is frowned upon by collectors and jewelers). It is the birth stone of March."
    },    
    "blue_sapphire": {
        "name": "Blue Sapphire",
        "img": "/blue_sapphire.jpg",
        "description": "Blue sapphire is a precious gemstone that is a variety of the mineral corundum, distinguished by its deep, rich blue color. The shade of blue can range from light to vivid dark blue, and the intensity often increases with the size of the stone. Sapphires are valued for their hardness, durability, and the lustrous shine that comes from their high refractive index."
    },
    "black_onyx": {
        "name": "Black Onyx",
        "img": "/onyx.jpg",
        "description": "Black Onyx is a striking variety of chalcedony, a type of microcrystalline quartz, known for its deep black color and smooth, glossy appearance. This gemstone has been valued for centuries for its captivating beauty and versatile applications. It is often used in jewelry-making, where it adds a touch of sophistication to rings, necklaces, bracelets, and earrings."
    },
    "citrine": {
        "name": "Citrine",
        "img": "/citrine.jpg",
        "description": "Citrine is a quartz variety that ranges in color from pale yellow to brownish-orange. It is prized for its warm, sunny colors that are often associated with positivity and joy. This gemstone is relatively durable and is commonly used in all types of jewelry, including rings, necklaces, and pendants."
    },
    "diamond": {
        "name": "Diamond",
        "img": "/diamond.jpg",
        "description": "Diamond is a solid form of the element carbon with its atoms arranged in a crystal structure called diamond cubic. Diamond has the highest hardness and thermal conductivity of any natural material, properties that are used in major industrial applications such as cutting and polishing tools. They are also the reason that diamond anvil cells can subject materials to pressures found deep in the Earth."
    },
    "emerald": {
        "name": "Emerald",
        "img": "/emerald.jpg",
        "description": "Emerald is a gemstone and a variety of the mineral beryl colored green by trace amounts of chromium or sometimes vanadium. Beryl has a hardness of 7.5–8 on the Mohs scale. Most emeralds have much material trapped inside during the gem's formation, so their toughness (resistance to breakage) is classified as generally poor."
    },
    "obsydian": {
        "name": "Obsydian",
        "img": "/obsydian.jpg",
        "description": "Obsidian is a naturally occurring volcanic glass formed when lava extruded from a volcano cools rapidly with minimal crystal growth. It is an igneous rock. Obsidian is produced from felsic lava, rich in the lighter elements such as silicon, oxygen, aluminium, sodium, and potassium. It is commonly found within the margins of rhyolitic lava flows known as obsidian flows."
    },
    "ruby": {
        "name": "Ruby",
        "img": "/ruby.jpg",
        "description": "Ruby is a pinkish red to blood-red colored gemstone, a variety of the mineral corundum (aluminium oxide). Ruby is one of the most popular traditional jewelry gems and is very durable. Other varieties of gem-quality corundum are called sapphires. Ruby is one of the traditional cardinal gems."
    },
    "turquoise": {
        "name": "Turquoise",
        "img": "/turquoise.jpg",
        "description": "Turquoise is an opaque, blue-to-green mineral that is a hydrous phosphate of copper and aluminium, with the chemical formula CuAl6(PO4)4(OH)8·4H2O. It is rare and valuable in finer grades and has been prized as a gemstone for millennia due to its hue. The robin egg blue or sky blue color of the Persian turquoise mined near the modern city of Nishapur, Iran, has been used as a guiding reference for evaluating turquoise quality."
    },
    "lapis_lazuli": {
        "name": "Lapis Lazuli",
        "img": "/lapis.jpg",
        "description": "Lapis lazuli, or lapis for short, is a deep-blue metamorphic rock used as a semi-precious stone that has been prized since antiquity for its intense color. Originating from the Persian word for the gem, lāžward, lapis lazuli is a rock composed primarily of the minerals lazurite, pyrite and calcite. "
    },
    "pink_sapphire": {
        "name": "Pink Sapphire",
        "img": "/pink_sapphire.jpg",
        "description": "Pink sapphire is another variety of the mineral corundum, tinged pink by traces of iron, titanium, or chromium. Its color ranges from soft pink to a deep magenta. Pink sapphires share the same properties as other sapphires, offering durability and vivid color, which makes them a popular choice for jewelry."
    },
    "quartz_clear": {
        "name": "Clear Quartz",
        "img": "/clear_quartz.jpg",
        "description": "Clear quartz, also known as crystal quartz or rock crystal, is a colorless, transparent variety of quartz that is highly valued for its clarity and crystalline beauty. It is often used in carvings, in jewelry, and as ornamental stones. Clear quartz is believed to have healing properties in many cultures."
    },
    "quartz_smoky": {
        "name": "Smoky Quartz",
        "img": "/quartz_smoky.jpg",
        "description": "Smoky quartz is a brown to black variety of quartz. Its color comes from free silicon formed from the natural irradiation of quartz. It offers transparency that ranges from almost completely transparent to an almost opaque brownish-gray or black crystal. Smoky quartz is used extensively as a gemstone in jewelry and for ornamental purposes."
    }
}

export const Results: React.FC<ResultsProps> = (props) => {
    const {data, src} = useData();
    const tab = [] 
    data.map((item) => {
        tab.push(predictionMapping[item])
    });
    console.log("tab",tab)

    return (
        <div className='flex flex-col'>
            <Navbar />
            <h1 className='text-4xl p-6 mb-6 text-center'>Here are your results!</h1>
            <Cards tab={tab}/>
            <button className='bg-darkAccent text-white p-4 mt-4 rounded-xl mx-auto'>
                <Link href='/predictions'>Back to Predictions!</Link>
            </button>

        </div>
    );
}
    

export default Results