import Link from 'next/link'
import React from 'react'

export default function Navbar() {
  return (
    <nav>
      <Link href="/">
        <h1 className="text-3xl py-4 bg-lightSecondary">CrystalClear</h1>
      </Link>
    </nav>
  )
}
