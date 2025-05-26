import 'package:flutter/material.dart';
import 'refractive_index_capture.dart';
import 'coating_capture.dart';
import 'composition_capture.dart';
import 'analysis_test_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Optical Authentication'),
        centerTitle: true,
        elevation: 0,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              colorScheme.surface,
              colorScheme.surface.withOpacity(0.8),
            ],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    'Select Analysis Type',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      letterSpacing: -0.5,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Choose the type of analysis you want to perform',
                    style: TextStyle(
                      fontSize: 16,
                      color: colorScheme.onSurface.withOpacity(0.7),
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 32),
                  _buildAnalysisCard(
                    context,
                    'Refractive Index Analysis',
                    'Analyze the refractive index of the lens',
                    Icons.lens,
                    colorScheme.primary,
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const RefractiveIndexCapture(),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  _buildAnalysisCard(
                    context,
                    'Coating Analysis',
                    'Analyze the lens coating and interference patterns',
                    Icons.filter_drama,
                    colorScheme.secondary,
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const CoatingCapture(),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  _buildAnalysisCard(
                    context,
                    'Composition Analysis',
                    'Analyze the lens material composition',
                    Icons.science,
                    colorScheme.tertiary,
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const CompositionCapture(),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  _buildAnalysisCard(
                    context,
                    'Gabor & Specular Flow Analysis',
                    'Texture and flow-based surface fingerprinting',
                    Icons.texture,
                    colorScheme.tertiary.withOpacity(0.7),
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const AnalysisTestScreen(),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  _buildAnalysisCard(
                    context,
                    'Quality Assessment',
                    'Comprehensive quality analysis',
                    Icons.assessment,
                    colorScheme.primary.withOpacity(0.7),
                    () {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Quality assessment coming soon!'),
                          behavior: SnackBarBehavior.floating,
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildAnalysisCard(
    BuildContext context,
    String title,
    String description,
    IconData icon,
    Color iconColor,
    VoidCallback onTap,
  ) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          padding: const EdgeInsets.all(20.0),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: iconColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  icon,
                  size: 32,
                  color: iconColor,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        letterSpacing: -0.5,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      description,
                      style: TextStyle(
                        color: Theme.of(context)
                            .colorScheme
                            .onSurface
                            .withOpacity(0.7),
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              ),
              Icon(
                Icons.arrow_forward_ios,
                size: 16,
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.5),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
