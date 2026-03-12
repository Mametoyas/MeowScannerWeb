'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { isUserLoggedIn } from '../../utils/auth';
import "../../styles/Login.css"
import Navbar from "@/components/login/Navbar"
import CatImage from "@/components/login/CatImage"
import LoginBox from "@/components/login/LoginBox"

export default function LoginPage() {
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
        <LoginBox />
      </div>
    </>
  )
}