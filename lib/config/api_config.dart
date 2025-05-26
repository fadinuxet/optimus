class ApiConfig {
  // Development server
  static const String devBaseUrl = 'http://192.168.5.81:8000';

  // Production server - replace with your actual server URL
  static const String prodBaseUrl = 'https://your-backend-server.com';

  // Use this to switch between dev and prod
  static const bool isProduction = true;

  // Get the current base URL
  static String get baseUrl => isProduction ? prodBaseUrl : devBaseUrl;
}
