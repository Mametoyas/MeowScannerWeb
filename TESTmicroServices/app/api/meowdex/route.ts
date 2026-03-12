import { NextRequest, NextResponse } from 'next/server';

// Mock data จาก meowdex_sheet
const meowdexData = [
  {
    CatID: "C0001",
    CatName: "Abyssinian",
    CatPersonal: "มีรูปร่างปราดเปรียว สง่างาม หูใหญ่แหลม และมีสีขนที่เป็นเอกลักษณ์เรียกว่า \"Ticked Coat\" (ขนหนึ่งเส้นมีหลายสีสลับกัน) ฉลาดมากและซนเหมือนลิง ชอบปีนป่าย",
    CatDetails: "เชื่อกันว่าเป็นหนึ่งในสายพันธุ์ที่เก่าแก่ที่สุดในโลก โดยมีรูปร่างคล้ายแมวในภาพวาดอียิปต์โบราณ แม้ชื่อจะมาจากประเทศอะบิสสิเนีย (เอธิโอเปียในปัจจุบัน) แต่หลักฐานทางพันธุกรรมบ่งชี้ว่าต้นกำเนิดจริงๆ อาจมาจากแถบชายฝั่งมหาสมุทรอินเดีย",
    Prices: "15,000 - 50,000",
    ImgURL: "https://petinsurance.com.au/wp-content/uploads/2016/07/Abyssinian_961x558-1.jpg"
  },
  {
    CatID: "C0002",
    CatName: "Bengal",
    CatPersonal: "ลายจุด (Rosettes) เหมือนเสือดาว มีความคล่องตัวสูง พลังงานล้นเหลือ และที่แปลกกว่าแมวทั่วไปคือ \"ชอบเล่นน้ำ\" มาก",
    CatDetails: "เกิดจากการผสมข้ามสายพันธุ์ระหว่าง แมวดาว (Asian Leopard Cat) กับแมวบ้าน เพื่อให้ได้แมวที่มีลักษณะเหมือนสัตว์ป่าแต่มีนิสัยเชื่องเหมือนแมวบ้าน เริ่มพัฒนาสายพันธุ์จริงจังในช่วงปี 1970-1980",
    Prices: "15,000 - 150,000",
    ImgURL: "https://www.trupanion.com/images/trupanionwebsitelibraries/bg/bengal-cat.jpg?sfvrsn=fc36dda4_5"
  },
  {
    CatID: "C0003",
    CatName: "Birman",
    CatPersonal: "มีฉายาว่า \"แมวศักดิ์สิทธิ์แห่งพม่า\" จุดเด่นคือมีถุงเท้าสีขาวบริสุทธิ์ที่เท้าทั้ง 4 ข้าง ตาสีฟ้าใส และขนยาวนุ่มสลวย นิสัยสุภาพ อ่อนโยน",
    CatDetails: "มีตำนานเล่าว่าเดิมทีเป็นแมวในวัดของพม่าที่คอยเฝ้าเทวรูปทองคำ เมื่อพระที่เป็นเจ้าของมรณภาพ แมวตัวนี้ก็ได้เปลี่ยนสีขนและสีตาตามพรของเทพธิดา แต่ในทางประวัติศาสตร์เริ่มเป็นที่รู้จักกว้างขวางเมื่อถูกนำเข้าไปยังฝรั่งเศสช่วงต้นศตวรรษที่ 20",
    Prices: "15,000 - 50,000",
    ImgURL: "https://www.thesprucepets.com/thmb/D5s03LINbIYpZuiG6uvBpKrAKXk=/3500x0/filters:no_upscale():strip_icc()/GettyImages-623368786-f66c97ad6d2d494287b448415f4340a8.jpg"
  }
];

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const catName = searchParams.get('catName');
    const catId = searchParams.get('catId');
    
    if (catName) {
      // ค้นหาตามชื่อแมว (case insensitive)
      const cat = meowdexData.find(cat => 
        cat.CatName.toLowerCase().includes(catName.toLowerCase())
      );
      
      if (cat) {
        return NextResponse.json({
          success: true,
          data: cat
        });
      }
    }
    
    if (catId) {
      // ค้นหาตาม CatID
      const cat = meowdexData.find(cat => cat.CatID === catId);
      
      if (cat) {
        return NextResponse.json({
          success: true,
          data: cat
        });
      }
    }
    
    // ถ้าไม่พบข้อมูล
    return NextResponse.json({
      success: false,
      message: 'Cat not found in meowdex'
    });
    
  } catch (error) {
    return NextResponse.json(
      { success: false, error: 'Failed to fetch meowdex data' },
      { status: 500 }
    );
  }
}