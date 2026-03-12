import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const { username, password } = await request.json();
  
  // Your Google Sheets logic here
  // This will be deployed on Vercel
  
  return NextResponse.json({ success: true });
}