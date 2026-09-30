package com.example.solarfarm.ml;

import org.springframework.stereotype.Service; import org.springframework.web.client.*; import org.springframework.http.*;
@Service public class MlClient {
 private final RestClient client; public MlClient(RestClient c){client=c;}
 public MlDtos.MlPrediction predict(MlDtos.MlTelemetry telemetry){ try { return client.post().uri("/api/v1/predict").contentType(MediaType.APPLICATION_JSON).body(telemetry).retrieve().body(MlDtos.MlPrediction.class); } catch (RestClientException e) { throw new MlServiceException("Prediction service is temporarily unavailable", e); } }
 public static class MlServiceException extends RuntimeException { public MlServiceException(String message, Throwable cause){super(message,cause);} }
}
