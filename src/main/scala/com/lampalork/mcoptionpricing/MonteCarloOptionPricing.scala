/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.lampalork.mcoptionpricing

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._ //import the implicit conversions
import org.apache.spark.SparkConf
import org.apache.spark.serializer.{KryoSerializer, KryoRegistrator}
import com.esotericsoftware.kryo.Kryo
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.distribution.MultivariateNormalDistribution

import scala.io.Source
import java.io.PrintWriter


class MyRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[Instrument])
  }
}

abstract class Option {
  def spot: Double
  def expiry: Double
  def sigma: Double
  def mu: Double
}

case class VanillaOption(spot: Double = 100.0, expiry: Double = 1.0, strike: Double = 100.0, sigma: Double = 0.10, mu: Double = 0.0)
  extends Option
{
  def payoff(xc: Array[Double]) = math.max(xc.last - strike,0)
  override def toString() = "VanillaOption with spot=" + spot + " strike=" + strike
}

case class UpAndOutOption(spot: Double = 100.0, expiry: Double = 1.0, strike: Double = 100.0, barrier: Double = 120.0, sigma: Double = 0.10, mu: Double = 0.0)
  extends Option
{
  def payoff(xc: Array[Double]) = {
    // reduceLeft is a concise/idiomatic approach
    // http://alvinalexander.com/scala/scala-reduceleft-examples
    var indicator = 1
    if (xc.reduceLeft(_ max _) > barrier) indicator = 0 // if barrier is crossed, barrier = 0
    
    indicator * math.max(xc.last - strike,0)
  }
  override def toString() = "UpAndOutOption with spot=" + spot + " strike=" + strike + " barrier=" + barrier
}



object MonteCarloOptionPricing {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Monte Carlo Option Pricing")
    sparkConf.set("spark.serializer", classOf[KryoSerializer].getName)
    sparkConf.set("spark.kryo.registrator", classOf[MyRegistrator].getName)
    val sc = new SparkContext(sparkConf)
    
    
    println("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    println("Monte Carlo Option Pricing")
    println("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    
    val numSimulations = args(0).toInt
    val parallelism = args(1).toInt    
    
    
    var option1 = new VanillaOption(100,1.0,100, 0.20, 0.05)
    var option2 = new UpAndOutOption(100,1.0,100, 120, 0.20, 0.05)
    var instruments = Array(option1,option2)
    
    // Send all instruments to every node
    val broadcastInstrument = sc.broadcast(option1)
    //val broadcastInstruments = sc.broadcast(instruments)
    
    //for (i <- 0 until broadcastInstruments.lenght)
    
     // Generate different seeds so that our simulations don't all end up with the same results
    val seed = System.currentTimeMillis()
    val seeds = (seed until seed + parallelism)
    val seedRdd = sc.parallelize(seeds, parallelism)
    
    // Main computation: run simulations and compute aggregate return for each
    val trialsRdd = seedRdd.flatMap(runSimulation(_, numSimulations / parallelism, broadcastInstrument.value))
    
    // Cache the results so that we don't recompute for both of the summarizations below
    trialsRdd.cache()
    
    // Calculate payoffs
    //val option1payoffs = trialsRdd.map((x: Array[Double]) => math.max(x.last - option1.strike,0))
    val option1payoffs = trialsRdd.map((x: Array[Double]) => option1.payoff(x))
    
    // Compute option price
    val option1price = math.exp(- option1.mu * option1.expiry) * option1payoffs.mean()
    println(option1 + ": " + option1price)
    
  }
  
  def runSimulation(seed: Long, numSimulations: Int, instrument: Option): Array[Array[Double]] = {
    //val rand = new MersenneTwister(seed)
    //val normal = new NormalDistribution(rand, 0.0, 1.0)
    val numTimeSteps = math.round( instrument.expiry / (1.0/365)).toInt    
    val dt = instrument.expiry / numTimeSteps
    
    val r = new scala.util.Random
    //val phi = Seq.fill(numSimulations * numTimeSteps)(r.nextGaussian)    
    
    val timeSeries = new Array[Array[Double]](numSimulations)
    
    // perform the numSimulations simulations
    for (i <- 0 until numSimulations) {
      
      // generate time serie
      val timeSerie = new Array[Double](numTimeSteps + 1)       
      timeSerie(0) = instrument.spot // init first stock price to spot   
      for (k <- 1 until (numTimeSteps+1)) {
        //val dX = normal.sample()
        val dX = r.nextGaussian()
        //val dX = phi(numSimulations*numTimeSteps + (k-1))
        val dS = instrument.mu * dt + instrument.sigma * math.sqrt(dt) * dX
        timeSerie(k) = timeSerie(k-1) * (1.0 + dS)
      }
      
      // save timeserie
      timeSeries(i) = timeSerie
    }
    
    timeSeries
  }
  
}
