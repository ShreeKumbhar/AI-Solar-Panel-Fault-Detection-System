package com.example.solarfarm.common;

import com.example.solarfarm.ml.MlClient.MlServiceException; import jakarta.servlet.http.HttpServletRequest; import org.springframework.http.*; import org.springframework.web.bind.annotation.*; import org.springframework.web.bind.MethodArgumentNotValidException; import java.time.Instant; import java.util.NoSuchElementException;
@RestControllerAdvice public class GlobalExceptionHandler {
 @ExceptionHandler(NoSuchElementException.class) ResponseEntity<Dtos.ErrorResponse> notFound(NoSuchElementException e,HttpServletRequest r){return error(404,"NOT_FOUND",e.getMessage(),r);}
 @ExceptionHandler(MlServiceException.class) ResponseEntity<Dtos.ErrorResponse> ml(MlServiceException e,HttpServletRequest r){return error(503,"ML_SERVICE_UNAVAILABLE",e.getMessage(),r);}
 @ExceptionHandler(MethodArgumentNotValidException.class) ResponseEntity<Dtos.ErrorResponse> validation(MethodArgumentNotValidException e,HttpServletRequest r){return error(400,"VALIDATION_ERROR","Request validation failed",r);}
 @ExceptionHandler(IllegalArgumentException.class) ResponseEntity<Dtos.ErrorResponse> badRequest(IllegalArgumentException e,HttpServletRequest r){return error(400,"INVALID_REQUEST",e.getMessage(),r);}
 @ExceptionHandler(Exception.class) ResponseEntity<Dtos.ErrorResponse> generic(Exception e,HttpServletRequest r){return error(500,"INTERNAL_ERROR","The request could not be completed",r);}
 private ResponseEntity<Dtos.ErrorResponse> error(int s,String e,String m,HttpServletRequest r){return ResponseEntity.status(s).body(new Dtos.ErrorResponse(Instant.now(),s,e,m,r.getRequestURI()));}
}
