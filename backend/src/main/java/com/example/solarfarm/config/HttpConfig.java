package com.example.solarfarm.config;

import org.springframework.context.annotation.*; import org.springframework.web.client.RestClient;
@Configuration public class HttpConfig { @Bean RestClient mlRestClient(org.springframework.core.env.Environment env) { return RestClient.builder().baseUrl(env.getProperty("ml.service.base-url","http://127.0.0.1:8000")).build(); } }
