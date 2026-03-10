import "../../styles/Login.css"
import Navbar from "@/components/login/Navbar"
import CatImage from "@/components/login/CatImage"
import LoginBox from "@/components/login/LoginBox"

export default function LoginPage() {
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