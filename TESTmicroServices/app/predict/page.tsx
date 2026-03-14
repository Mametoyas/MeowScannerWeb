"use client"
import { useState, useEffect } from "react";
import "../../styles/Predict.css";
import PredictImage from "@/components/predict/PredictImage";
import PredictResult from "@/components/predict/PredictResult";
import { PredictionData } from "@/components/predict/types/predict";
import Navbar from "@/components/predict/Navbar";
import { modelService, dataService, meowdexService } from "@/components/api";
import { authUtils } from "@/components/auth";
import ProtectedRoute from "@/components/ProtectedRoute";

interface LocationData {
  lat: number;
  lon: number;
}

interface PredictMapProps {
  location: LocationData;
  catName: string;
  isFromImage?: boolean;
}

function PredictMap({ location, catName, isFromImage = false }: PredictMapProps) {
  const [mapLoaded, setMapLoaded] = useState(false);

  useEffect(() => {
    if (!mapLoaded) {
      // Load Leaflet CSS
      const cssLink = document.createElement('link');
      cssLink.rel = 'stylesheet';
      cssLink.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
      document.head.appendChild(cssLink);

      // Load Leaflet JS
      const script = document.createElement('script');
      script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
      script.onload = () => {
        const L = (window as any).L;
        
        // Create map
        const map = L.map('predict-map').setView([location.lat, location.lon], 15);
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        // Add marker
        L.marker([location.lat, location.lon])
          .addTo(map)
          .bindPopup(`
            <div style="text-align: center; padding: 5px;">
              <h4 style="margin: 0 0 5px 0; color: #333;">🐱 ${catName}</h4>
              <p style="margin: 2px 0; font-size: 12px; color: #666;">${isFromImage ? 'From image EXIF data' : 'From current GPS location'}</p>
              <a href="https://www.google.com/maps?q=${location.lat},${location.lon}" 
                 target="_blank" 
                 style="color: #1a73e8; text-decoration: none; font-size: 12px;">
                📍 View in Google Maps
              </a>
            </div>
          `)
          .openPopup();

        setMapLoaded(true);
      };
      
      document.head.appendChild(script);
    }
  }, [location, catName, isFromImage, mapLoaded]);

  return (
    <div className="predict-map-container">
      <h3>📍 Cat Location {isFromImage ? '(From Image)' : '(Current GPS)'}</h3>
      <div id="predict-map" style={{ height: '300px', width: '100%', borderRadius: '0.5rem' }}></div>
      <div className="location-info">
        <p>📍 GPS: {location.lat.toFixed(8)}, {location.lon.toFixed(8)}</p>
        <button 
          onClick={() => window.open(`https://www.google.com/maps?q=${location.lat},${location.lon}`, '_blank')}
          className="google-maps-btn"
        >
          🗺️ Open in Google Maps
        </button>
      </div>
    </div>
  );
}

export default function PredictPage() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [locationData, setLocationData] = useState<LocationData | null>(null);
  const [currentLocation, setCurrentLocation] = useState<LocationData | null>(null);
  const [gpsPermission, setGpsPermission] = useState<'granted' | 'denied' | 'prompt' | null>(null);

  useEffect(() => {
    // Check GPS permission on load
    if (navigator.geolocation) {
      navigator.permissions?.query({name: 'geolocation'}).then((result) => {
        setGpsPermission(result.state as any);
      }).catch(() => {
        setGpsPermission('prompt');
      });
    }
  }, []);

  const requestGPSPermission = () => {
    if (!navigator.geolocation) {
      setMessage('GPS not supported by this browser');
      return;
    }

    setMessage('Requesting high-accuracy GPS...');
    
    // Try to get multiple readings for better accuracy
    let readings = [];
    let readingCount = 0;
    const maxReadings = 3;
    
    const getReading = () => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          readingCount++;
          readings.push({
            lat: position.coords.latitude,
            lon: position.coords.longitude,
            accuracy: position.coords.accuracy,
            timestamp: position.timestamp
          });
          
          console.log(`GPS Reading ${readingCount}:`, {
            lat: position.coords.latitude,
            lon: position.coords.longitude,
            accuracy: position.coords.accuracy + 'm',
            timestamp: new Date(position.timestamp).toLocaleTimeString()
          });
          
          if (readingCount >= maxReadings) {
            // Use the most accurate reading
            const bestReading = readings.reduce((best, current) => 
              current.accuracy < best.accuracy ? current : best
            );
            
            console.log('Best GPS Reading:', bestReading);
            
            const location = {
              lat: bestReading.lat,
              lon: bestReading.lon
            };
            
            setCurrentLocation(location);
            setGpsPermission('granted');
            setMessage(`GPS enabled! Best accuracy: ${bestReading.accuracy.toFixed(1)}m | Location: ${location.lat.toFixed(8)}, ${location.lon.toFixed(8)}`);
          } else {
            // Get another reading
            setTimeout(getReading, 1000);
          }
        },
        (error) => {
          console.error('GPS Error:', error);
          setGpsPermission('denied');
          switch(error.code) {
            case error.PERMISSION_DENIED:
              setMessage('GPS permission denied. Please enable location access in browser settings.');
              break;
            case error.POSITION_UNAVAILABLE:
              setMessage('GPS location unavailable. Try using mobile device for better accuracy.');
              break;
            case error.TIMEOUT:
              setMessage('GPS request timeout. Try again or use mobile device.');
              break;
            default:
              setMessage('GPS error occurred.');
              break;
          }
        },
        {
          enableHighAccuracy: true,
          timeout: 20000,
          maximumAge: 0
        }
      );
    };
    
    getReading();
  };

  const handleImageUpload = (imageFile: File) => {
    setSelectedImage(imageFile);
    setPrediction(null);
    setLocationData(null);
    setMessage('Image selected. Click Predict to analyze.');
  };

  const handlePredict = async () => {
    if (!selectedImage) return;
    
    setLoading(true);
    setMessage('Processing image...');
    
    try {
      // Call Model API for prediction
      const result = await modelService.predict(selectedImage);
      
      console.log('Prediction result from model:', result);
      
      // Get cat info from meowdex if breed is detected
      let catInfo = null;
      if (result.detections?.[0] && result.detections[0].class !== "ไม่เจอแมว") {
        try {
          console.log('Fetching cat info for:', result.detections[0].class);
          const meowdexResult = await meowdexService.getCatInfo(result.detections[0].class);
          console.log('Meowdex result:', meowdexResult);
          if (meowdexResult.success) {
            catInfo = meowdexResult.data;
            console.log('Cat info found:', catInfo);
          } else {
            console.log('Cat info not found in database');
          }
        } catch (error) {
          console.error('Failed to fetch meowdex data:', error);
        }
      }
      
      const predictionData: PredictionData = {
        imageUrl: `data:image/jpeg;base64,${result.imagedetect}`,
        catCount: result.detections?.[0]?.class === "ไม่เจอแมว" ? 0 : (result.bboxes?.length || result.detections?.length || 0),
        breed: result.detections?.[0]?.class || "Unknown",
        features: catInfo?.CatPersonal || (result.detections?.[0]?.class === "ไม่เจอแมว" ? "ไม่พบแมวในภาพ" : ""),
        confidence: (result.detections?.[0]?.conf * 100 || 0),
        catPersonal: catInfo?.CatPersonal || null,
        catDetails: catInfo?.CatDetails || null,
        prices: catInfo?.Prices || null,
        imgURL: catInfo?.ImgURL || null
      };
      
      setPrediction(predictionData);
      
      // Save to database
      const user = authUtils.getCurrentUser();
      console.log('Current user:', user);
      
      if (user && result.detections?.[0] && result.detections[0].class !== "ไม่เจอแมว") {
        try {
          console.log('Saving history...', user.user_id, result.detections[0]);
          
          // Save history
          const historyResult = await dataService.addHistory(user.user_id, result.detections[0].cat_id || result.detections[0].class);
          console.log('History result:', historyResult);
          
          // Save location if GPS found in image or use current location
          let finalLocation = result.location || currentLocation;
          
          if (finalLocation) {
            console.log('Saving location...', finalLocation);
            setLocationData(finalLocation);
            const locationResult = await dataService.addMapLocation(
              user.user_id,
              finalLocation.lon,
              finalLocation.lat,
              result.detections[0].cat_id || result.detections[0].class
            );
            console.log('Location result:', locationResult);
            
            const locationSource = result.location ? 'from image EXIF' : 'from current GPS';
            setMessage(`Prediction completed! GPS ${locationSource}: ${finalLocation.lat.toFixed(4)}, ${finalLocation.lon.toFixed(4)}`);
          } else {
            console.log('No GPS data found');
            setLocationData(null);
            setMessage('Prediction completed! No GPS data available. Enable GPS to save location.');
          }
          
        } catch (error) {
          console.error('Failed to save data:', error);
          setMessage('Prediction completed! (Failed to save to database)');
        }
      } else {
        console.log('No user, no detections, or no cat found:', { 
          user: !!user, 
          detections: result.detections,
          firstDetection: result.detections?.[0]
        });
        setMessage('Prediction completed! (Not logged in, no detections, or no cat found)');
      }
      
    } catch (error) {
      console.error('Prediction failed:', error);
      setMessage('Prediction failed, model API gateway error.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ProtectedRoute>
      <Navbar />
      <div className="predict-container">
        {message && (
          <div className={`message-bar ${message.includes('completed') ? 'success' : message.includes('failed') ? 'error' : ''}`}>
            {message}
          </div>
        )}
        <div className="main-layout">
          <PredictImage 
            imageUrl={prediction?.imageUrl || (selectedImage ? URL.createObjectURL(selectedImage) : undefined)}
            onImageUpload={handleImageUpload}
            onPredict={handlePredict}
            loading={loading}
          />
          <div className="result-section">
            <PredictResult data={prediction} loading={loading} />
            
            {/* GPS Control */}
            <div className="gps-control">
              {gpsPermission === 'granted' && currentLocation ? (
                <div className="gps-status granted">
                  <p>GPS Enabled</p>
                  <p className="coordinates">{currentLocation.lat.toFixed(8)}, {currentLocation.lon.toFixed(8)}</p>
                  <div className="gps-actions">
                    <button 
                      onClick={requestGPSPermission}
                      className="refresh-gps-btn"
                      disabled={loading}
                    >
                      Refresh GPS
                    </button>
                    <button 
                      onClick={() => window.open(`https://www.google.com/maps?q=${currentLocation.lat},${currentLocation.lon}`, '_blank')}
                      className="view-location-btn"
                    >
                      View Location
                    </button>
                  </div>
                </div>
              ) : (
                <div className="gps-status">
                  <button 
                    onClick={requestGPSPermission}
                    className="gps-btn"
                    disabled={loading}
                  >
                    Enable High-Accuracy GPS
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {locationData && prediction && (
          <div className="map-section">
            <PredictMap 
              location={locationData} 
              catName={prediction.breed}
              isFromImage={!!locationData && locationData !== currentLocation}
            />
          </div>
        )}
      </div>
    </ProtectedRoute>
  );
}