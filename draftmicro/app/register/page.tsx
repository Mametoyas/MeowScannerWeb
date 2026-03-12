'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { isUserLoggedIn } from '../../utils/auth';
import "../../styles/Register.css"
import Navbar from "@/components/register/Navbar"
import CatImage from "@/components/register/CatImage"
import RegisterBox from "@/components/register/RegisterBox"

export default function RegisterPage() {
  const router = useRouter();

  useEffect(() => {
    if (isUserLoggedIn()) {
      router.push('/main');
    }
  }, [router]);

  return (
    <>
      <Navbar />

      <div className="container">
        <CatImage />
        <RegisterBox />
      </div>
    </>
  )
}